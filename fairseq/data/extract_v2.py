# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from fairseq.data import data_utils
from fairseq.data.encoders.gpt2_bpe import get_encoder

class ExtractDatasetV2(torch.utils.data.Dataset):
    def __init__(
        self,
        src_dataset,
        src_dict,
        seed,
        max_positon=1024,
        **kwargs
    ):
        self.src_dataset = src_dataset
        self.src_dict = src_dict
        self.seed = seed
        # self.extract_num = extract_num
        # bpe = get_encoder('encoder.json', 'vocab.bpe')
        self.dot_token = [4, 479]
        self.max_positon = max_positon

    def __getitem__(self, index):
        with data_utils.numpy_seed(self.seed + index):
            choices = np.arange(3)
            choice = np.random.choice(choices)
            if choice == 0:
                return self.extract_sents(index, 'all')
            elif choice == 1:
                return self.extract_sents(index, 3)
            else :
                return self.extract_sents(index, 'half')
    
    def extract_sents(self, index, extract_num):
        assert isinstance(extract_num, int) or extract_num == 'half' or extract_num == 'all'
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
        if not extract_num == 'all': 
            select_idx = np.sort(np.random.permutation(range(len(src_sentences)))[: extract_num])
        else:
            select_idx = np.sort(np.random.permutation(range(len(src_sentences)))[: len(src_sentences) // 2])
        try:
            if not extract_num == 'all':
                select_sents = [src_sentences[idx] for idx in select_idx] + [np.array([self.src_dict.eos()], dtype=src_sentences[0].dtype)]
                src_tokens = torch.from_numpy(np.concatenate(select_sents))
            else:
                src_tokens = self.src_dataset[index]

            tgt_sents = []
            select_pos = []
            visible_pos = torch.BoolTensor(self.max_positon)
            visible_pos.fill_(False)
            pre_len = 0
            for idx, sent in enumerate(src_sentences):
                tgt_sents += [sent]
                cur_len = sum([len(i) for i in tgt_sents])
                # tgt_sents += ([sent] + [np.array([self.src_dict.index('<sep>')], dtype=sent.dtype)])
                if idx in select_idx :
                    select_pos.append([pre_len, cur_len])
                    visible_pos[pre_len: cur_len] = True
                    # if idx + 1 in select_idx:
                    #     p = np.random.uniform(0, 1)
                    #     if p < 0.1:
                    #         tgt_sents += [np.array([self.src_dict.index('<sep>')], dtype=sent.dtype)]
                    #         cur_len += 1
                # elif idx not in select_idx and idx + 1 in select_idx and idx < len(src_sentences):
                #     tgt_sents += [np.array([self.src_dict.index('<sep>')], dtype=sent.dtype)]
                #     cur_len += 1
                pre_len = cur_len

            tgt_sents += [np.array([self.src_dict.eos()], dtype=src_sentences[0].dtype)]
            tgt_tokens = torch.from_numpy(np.concatenate(tgt_sents))
            if len(tgt_tokens) > self.max_positon:
                tgt_tokens = torch.cat([tgt_tokens[: self.max_positon-1], torch.LongTensor([self.src_dict.eos()])])
            visible_pos = visible_pos[:len(tgt_tokens)]
            visible_pos[-1] = True

        except IndexError:
            if not select_idx:
                tgt_tokens = torch.cat([self.src_dataset[index], torch.LongTensor([self.src_dict.eos()])])
                return {
                "src_tokens": self.src_dataset[index],
                "tgt_tokens": tgt_tokens,
                "visible_pos": tgt_tokens != self.src_dict.pad()
            }
            else:
                print(select_idx)
                print(len(src_sentences))
                print(src_sentences)
                raise RuntimeError('select idx: {}, len src_sentence: {}, src_sentence: {}'.format(select_idx, len(src_sentences), src_sentences))
        return {
            "src_tokens": src_tokens,
            "tgt_tokens": tgt_tokens,
            "visible_pos": visible_pos
        }

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
