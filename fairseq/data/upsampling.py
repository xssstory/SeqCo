# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from fairseq.data import data_utils
from fairseq.data.encoders.gpt2_bpe import get_encoder

class UpsamplingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        src_dataset,
        candidate_dataset,
        src_dict,
        seed,
        # extract_num,
        **kwargs
    ):
        self.src_dataset = src_dataset
        self.candidate_dataset = candidate_dataset
        self.src_dict = src_dict
        self.seed = seed
        # self.extract_num = extract_num
        # bpe = get_encoder('encoder.json', 'vocab.bpe')
        self.dot_token = [4, 479]

    def __getitem__(self, index):
        with data_utils.numpy_seed(self.seed + index):
            choice = np.random.randint(2)
            if choice == 0:
                return self.src_dataset[index]
            elif choice == 1:
                return self.expand_sents(index, 1)
    
    def expand_sents(self, index, expand_length):

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

        expand_length = len(src_sentences) * expand_length
        candidate_sents = []
        for idx in range(expand_length):
            while True:
                selected_doc = np.random.randint(len(self.candidate_dataset))
                selected_doc = self.candidate_dataset[selected_doc].numpy()
                selected_sents = np.split(selected_doc, np.where(np.bitwise_or(selected_doc==self.dot_token[0], selected_doc==self.dot_token[1]))[0]+1)
                if len(selected_sents) <= 5:
                    continue
                try:
                    selected_sent = np.random.choice(selected_sents)
                except:
                    raise RuntimeError('{}:{}'.format(type(selected_sents), selected_sents))
                if len(selected_sent) != 1:
                    break
            if selected_sent[-1] == self.src_dict.eos():
                selected_sent = selected_sent[:-1]
            candidate_sents.append(selected_sent)
        
        out_sents = []
        while src_sentences and candidate_sents:
            if np.random.randint(2) == 0:
                out_sents.append(src_sentences.pop(0))
            else:
                out_sents.append(candidate_sents.pop(0))
        if src_sentences:
            out_sents.extend(src_sentences)
        if candidate_sents:
            out_sents.extend(candidate_sents)

        out_sents += [np.array([self.src_dict.eos()], dtype=out_sents[0].dtype)]
        out_tokens = torch.from_numpy(np.concatenate(out_sents))
        return out_tokens

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
