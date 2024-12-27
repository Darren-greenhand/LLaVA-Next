"""
action_tokenizer.py

Extension class; wraps base LLM/VLM tokenizer with logic to discretize and tokenize continuous robot actions.
"""

from typing import List, Union

import numpy as np
from transformers import PreTrainedTokenizerBase


# class ActionTokenizer:
#     def __init__(
#         self, tokenizer: PreTrainedTokenizerBase, bins: int = 256, min_action: int = -1, max_action: int = 1
#     ) -> None:
#         """
#         Discretizes continuous robot actions into N bins per dimension and maps to the least used tokens.

#         NOTE =>> by default, assumes a BPE-style tokenizer akin to the LlamaTokenizer, where *the least used tokens*
#                  appear at the end of the vocabulary!

#         :param tokenizer: Base LLM/VLM tokenizer to extend.
#         :param bins: Number of bins for each continuous value; we'll adopt a uniform binning strategy.
#         :param min_action: Minimum action value (for clipping, setting lower bound on bin interval).
#         :param max_action: Maximum action value (for clipping, setting upper bound on bin interval).
#         """
#         self.tokenizer, self.n_bins, self.min_action, self.max_action = tokenizer, bins, min_action, max_action

#         # Create Uniform Bins + Compute Bin Centers
#         self.bins = np.linspace(min_action, max_action, self.n_bins)
#         self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

#         # [Contract] Set "action_token_begin_idx" based on `self.tokenizer.vocab_size - (self.n_bins + 1)`
#         #   =>> Assumes we're always overwriting the final `n_bins` tokens of the vocabulary!
#         self.action_token_begin_idx: int = int(self.tokenizer.vocab_size - (self.n_bins + 1))

#     def __call__(self, action: np.ndarray) -> Union[str, List[str]]:
#         """Clip & bin actions to *the last `n_bins` tokens* of the vocabulary (e.g., tokenizer.vocab[-256:])."""
#         action = np.clip(action, a_min=float(self.min_action), a_max=float(self.max_action))
#         discretized_action = np.digitize(action, self.bins)

#         # Handle single element vs. batch
#         if len(discretized_action.shape) == 1:
#             return self.tokenizer.decode(list(self.tokenizer.vocab_size - discretized_action))
#         else:
#             return self.tokenizer.batch_decode((self.tokenizer.vocab_size - discretized_action).tolist())

#     def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
#         """
#         Returns continuous actions for discrete action token IDs.

#         NOTE =>> Because of the way the actions are discretized w.r.t. the bins (and not the bin centers), the
#                  digitization returns bin indices between [1, # bins], inclusive, when there are actually only
#                  (# bins - 1) bin intervals.

#                  Therefore, if the digitization returns the last possible index, we map this to the last bin interval.

#         EXAMPLE =>> Let's say self._bins has 256 values. Then self._bin_centers has 255 values. Digitization returns
#                     indices between [1, 256]. We subtract 1 from all indices so that they are between [0, 255]. There
#                     is still one index (i==255) that would cause an out-of-bounds error if used to index into
#                     self._bin_centers. Therefore, if i==255, we subtract 1 from it so that it just becomes the index of
#                     the last bin center. We implement this simply via clipping between [0, 255 - 1].
#         """
#         discretized_actions = self.tokenizer.vocab_size - action_token_ids
#         discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)

#         return self.bin_centers[discretized_actions]

#     @property
#     def vocab_size(self) -> int:
#         return self.n_bins
    
class ActionTokenizer:
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, bins: int = 256, min_action: int = -1, max_action: int = 1
    ) -> None:
        self.tokenizer, self.n_bins, self.min_action, self.max_action = tokenizer, bins, min_action, max_action

        # print('initializing action tokenizer, the origin len(tokenizer) is:', len(self.tokenizer))
        self.origin_len = len(self.tokenizer)
        # 创建等间距的区间和计算区间中心
        self.bins = np.linspace(min_action, max_action, self.n_bins + 1)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # 添加新的动作 tokens 到 tokenizer
        self.action_tokens = [f"<ACTION_{i}>" for i in range(self.n_bins)]
        self.tokenizer.add_tokens(self.action_tokens)
        self.action_token_ids = self.tokenizer.convert_tokens_to_ids(self.action_tokens)
        self.new_len = len(self.tokenizer)

        # print('initializing action tokenizer, the new len(tokenizer) is:', len(self.tokenizer))

    def __call__(self, action: np.ndarray) -> Union[str, List[str]]:


        action = np.clip(action, a_min=float(self.min_action), a_max=float(self.max_action))
        discretized_action = np.digitize(action, self.bins) - 1
        discretized_action = np.clip(discretized_action, a_min=0, a_max=self.n_bins - 1)

        action_token_ids = np.array(self.action_token_ids)[discretized_action]

        if len(action_token_ids.shape) == 1:
            # return self.tokenizer.decode(action_token_ids)
            return action_token_ids
        else:
            # return self.tokenizer.batch_decode(action_token_ids.tolist())
            return [ids.tolist() for ids in action_token_ids]

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        token_id_to_action_index = {tid: idx for idx, tid in enumerate(self.action_token_ids)}
        vectorized_lookup = np.vectorize(token_id_to_action_index.get)
        action_indices = vectorized_lookup(action_token_ids)

        return self.bin_centers[action_indices]

    # 返回所有 tokens 的数量，包括动作 tokens
    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    @property
    def num_new_tokens(self) -> int:
        return self.new_len - self.origin_len

    @property
    def action_vocab_size(self) -> int:
        return len(self.action_tokens)    