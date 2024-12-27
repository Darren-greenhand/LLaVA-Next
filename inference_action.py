# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.actiontokenizer import ActionTokenizer

from PIL import Image
import requests
import copy
import torch

import sys
import warnings

warnings.filterwarnings("ignore")
pretrained = "/data/jcy/ckpt/llavaov-finetune/llava-openvla-kuka_33_franctal_33_bridge_33"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)  # Add any other thing you want to pass in llava_model_args
actiontokenizer = ActionTokenizer(tokenizer)

print(model.model.embed_tokens.weight.shape)
print(model.lm_head.weight.shape)

model.eval()

# url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
image_path = '/data/jcy/project/LLaVA-NeXT/llava_v1_5_radar.jpg'
# image = Image.open(requests.get(url, stream=True).raw)
image = Image.open(image_path)
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
question = DEFAULT_IMAGE_TOKEN + "\nWhat action should take?"
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [image.size]


cont = model.generate(
    input_ids,
    images=image_tensor,
    image_sizes=image_sizes,
    do_sample=False,
    temperature=0,
    max_new_tokens=4096,
)

print(cont)
# tokenizer 用于解码范围在 151647 - 151903 之外的 token
# actiontokenizer 用于解码范围在 151647 - 151903 之内的 token

token_ids = cont.cpu().tolist()[0]
decoded_text = []

for token_id in token_ids:
    if 151647 <= token_id <= 151903:
        decoded_token = actiontokenizer.decode_token_ids_to_actions([token_id])
    else:
        decoded_token = tokenizer.decode([token_id])
    decoded_text.append(decoded_token)
final_text = ''.join(str(item) for item in decoded_text)

print(final_text)


# text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
# print(text_outputs)
