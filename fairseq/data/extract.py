# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from fairseq.data import data_utils
from fairseq.data.encoders.gpt2_bpe import get_encoder

class ExtractDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        src_dataset,
        src_dict,
        seed,
        insert_sep=False,
        # extract_num,
        **kwargs
    ):
        self.src_dataset = src_dataset
        self.src_dict = src_dict
        self.seed = seed
        # self.extract_num = extract_num
        # bpe = get_encoder('encoder.json', 'vocab.bpe')
        self.dot_token = [4, 479]
        self.insert_sep = insert_sep
        self.sep_idx = src_dict.index('<sep>')
        self.max_positon = 1024

    def __getitem__(self, index):
        with data_utils.numpy_seed(self.seed + index):
            choices = np.arange(3)
            choice = np.random.choice(choices)
            if choice == 0:
                src_tokens = self.src_dataset[index]
            elif choice == 1:
                src_tokens = self.extract_sents(index, 3)
            else :
                src_tokens = self.extract_sents(index, 'half')
        return src_tokens if not self.insert_sep else {'src_tokens': src_tokens, 'fix_idx': self.sep_idx}
    
    def extract_sents(self, index, extract_num):
        assert isinstance(extract_num, int) or extract_num == 'half'
        src_sentences = self.src_dataset[index].numpy()
        src_sentences = np.split(src_sentences, np.where(np.bitwise_or(src_sentences==self.dot_token[0], src_sentences==self.dot_token[1]))[0]+1)
        if len(src_sentences[-1]) == 1:
            if not src_sentences[-1].item() == self.src_dict.eos():
                raise TypeError('The last token mast be </s> !')
            else:
                del src_sentences[-1]
        else:
            assert src_sentences[-1][-1] == self.src_dict.eos()
            src_sentences[-1] = src_sentences[-1][:-1]
        if extract_num == 'half':
            extract_num = len(src_sentences) // 2
        select_idx = np.sort(np.random.permutation(range(len(src_sentences)))[: extract_num])
        try:
            if not self.insert_sep:
                select_sents = [src_sentences[idx] for idx in select_idx] + [np.array([self.src_dict.eos()], dtype=src_sentences[0].dtype)]
            else:
                select_sents = [np.append(src_sentences[idx], self.sep_idx) for idx in select_idx] + [np.array([self.src_dict.eos()], dtype=src_sentences[0].dtype)]
            src_tokens = torch.from_numpy(np.concatenate(select_sents))
            if len(src_tokens) > self.max_positon:
                src_tokens = torch.cat([src_tokens[: self.max_positon-1], torch.LongTensor([self.src_dict.eos()])])
        except IndexError:
            if not select_idx:
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
