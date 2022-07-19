# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from fairseq.data import data_utils
from fairseq.data.encoders.gpt2_bpe import get_encoder

class BartDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        src_dataset,
        src_dict,
        seed,
        mask_probability = 0.35,
        mask_full_sent = False,
        **kwargs
    ):
        self.src_dataset = src_dataset
        self.src_dict = src_dict
        self.seed = seed
        # self.extract_num = extract_num
        # bpe = get_encoder('encoder.json', 'vocab.bpe')
        self.dot_token = [4, 479]
        self.span_idx = src_dict.index('<sep>')
        self.max_positon = 1024
        self.mask_probability = mask_probability
        self.mask_full_sent = mask_full_sent
        self.perm_sent = not mask_full_sent

    def __getitem__(self, index):
        with data_utils.numpy_seed(self.seed + index):
            src_tokens = self.preprocess_sents(index)
        return src_tokens
    
    def preprocess_sents(self, index):
        # assert isinstance(extract_num, int) or extract_num == 'half'
        src_sentences = self.src_dataset[index].numpy()
        src_sentences = np.split(src_sentences, np.where(np.bitwise_or(src_sentences==self.dot_token[0], src_sentences==self.dot_token[1]))[0]+1)
        if len(src_sentences) == 0:
            return self.src_dataset[index]
        if len(src_sentences[-1]) == 1:
            if not src_sentences[-1].item() == self.src_dict.eos():
                raise TypeError('The last token mast be </s> !')
            else:
                del src_sentences[-1]
        else:
            assert src_sentences[-1][-1] == self.src_dict.eos()
            src_sentences[-1] = src_sentences[-1][:-1]

        # replace span tokens with <sep>
        new_src_sentences = []
        for idx, src_sentence in enumerate(src_sentences):
            new_src_sentence = src_sentence.copy()

            if self.mask_full_sent:
                if np.random.randint(100) < self.mask_probability * 100:
                    new_src_sentence = np.array([self.span_idx], dtype=src_sentence.dtype)
                else:
                    if np.random.randint(100 - self.mask_probability * 100) < self.mask_probability * 100:
                        new_src_sentence = np.append(new_src_sentence, self.span_idx)
            else:
                sent_len = len(src_sentence)
                num_success = 0
                num_fail = 0
                while (len(new_src_sentence) - num_success) > sent_len * (1 - self.mask_probability):
                    if num_fail > 10:
                        break
                    if len(new_src_sentence) <= 1:
                        break
                    span_len = np.random.poisson(3)
                    start = np.random.randint(len(new_src_sentence) - 1)
                    end = min(start + span_len, len(new_src_sentence) - 2)
                    if self.span_idx in new_src_sentence[start: end]:
                        num_fail += 1
                        continue
                    new_src_sentence = np.concatenate([np.append(new_src_sentence[:start], self.span_idx), new_src_sentence[end:]])
                    num_success += 1
            if idx > 0 and np.array_equal(np.array([self.span_idx], dtype=src_sentence.dtype), new_src_sentences[-1]) and \
            np.array_equal(np.array([self.span_idx], dtype=src_sentence.dtype), new_src_sentence):
                continue
            else:
                new_src_sentences.append(new_src_sentence)
        src_sentences = new_src_sentences


        perm_idx = np.random.permutation(range(len(src_sentences))) if self.perm_sent else range(len(src_sentences))

        try:

            perm_sents = [src_sentences[idx] for idx in perm_idx] + [np.array([self.src_dict.eos()], dtype=src_sentences[0].dtype)]
            src_tokens = torch.from_numpy(np.concatenate(perm_sents))
            if len(src_tokens) > self.max_positon:
                src_tokens = torch.cat([src_tokens[: self.max_positon-1], torch.LongTensor([self.src_dict.eos()])])
        except IndexError:
            if not perm_idx:
                src_tokens = self.src_dataset[index]
            else:
                print(select_idx)
                print(len(src_sentences))
                print(src_sentences)
                raise RuntimeError('select idx: {}, len src_sentence: {}, src_sentence: {}'.format(select_idx, len(src_sentences), src_sentences))
        return src_tokens

    def __len__(self):
        """
        The length of the noising dataset is the length of src.
        """
        return len(self.src_dataset)

    @property
    def sizes(self):
        return self.src_dataset.sizes

    @property
    def supports_prefetch(self):
        return self.src_dataset.supports_prefetch

    def prefetch(self, indices):
        if self.src_dataset.supports_prefetch:
            self.src_dataset.prefetch(indices)
