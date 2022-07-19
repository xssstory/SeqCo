# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import math

from fairseq import utils
from fairseq.data import data_utils
from fairseq.data.backsum_pair_dataset import LANG_TOKEN_DIC
from . import FairseqDataset
import numpy as np

PAD_IDX = 1
EOS_IDX = 2

def preprocess_samples(samples, lang, max_generate_len):
    dot_token = [4, 479]
    if lang == 'cnndm':
        return  samples
    elif lang == 'inshorts':
        for sample in samples:
            src_sentences = sample['source'].numpy()
            src_sentences = np.split(src_sentences, np.where(np.bitwise_or(src_sentences==dot_token[0], src_sentences==dot_token[1]))[0]+1)
            if len(src_sentences[-1]) == 1:
                if not src_sentences[-1].item() == EOS_IDX:
                    raise TypeError('The last token mast be </s> !')
                else:
                    del src_sentences[-1]
            else:
                assert src_sentences[-1][-1] == EOS_IDX
            if src_sentences[0][0] == EOS_IDX:
                src_sentences[0] = src_sentences[0][1:]
            
            avg_len = (max_generate_len - len(sample['source'])) // len(src_sentences)

            tgt_sents = []
            select_pos = []
            visible_pos = torch.BoolTensor(max_generate_len)
            visible_pos.fill_(False)
            pre_len = 0
            with data_utils.numpy_seed(sample['id']):
                for idx, sent in enumerate(src_sentences):
                    tgt_sents += [sent]
                    cur_len = sum([len(i) for i in tgt_sents])
                    select_pos.append([pre_len, cur_len])
                    visible_pos[pre_len: cur_len] = True  
                    if np.random.randint(2) == 0:
                        tgt_sents += [[PAD_IDX for _ in range(avg_len)]]
                    else:
                        tgt_sents += [[PAD_IDX for _ in range(len(sent))]]
                    pre_len = sum([len(i) for i in tgt_sents])

            tgt_sents += [np.array([EOS_IDX], dtype=src_sentences[0].dtype)]
            tgt_tokens = torch.from_numpy(np.concatenate(tgt_sents))
            visible_pos = visible_pos[:len(tgt_tokens)]
            visible_pos[-1] = True
            sample['target'] = tgt_tokens
            sample['visible_pos'] = visible_pos
        return samples

def backtranslate_samples(samples, collate_fn, generate_fn, accumulate_step, lang, max_generate_len, cuda=True):

    with torch.no_grad():
        samples = utils.move_to_cpu(samples)
        samples = preprocess_samples(samples, lang, max_generate_len)
        collated_samples = collate_fn(samples)
        collated_samples = utils.move_to_cuda(collated_samples) if cuda else collated_samples
        generated_sources = generate_fn(collated_samples)

    id_to_src = {
        sample['id']: sample['source'] for sample in samples
    }
    # bsz = len(generated_sources) // accumulate_step
    bsz = math.ceil(len(generated_sources) / accumulate_step)
    src_tokens_list = torch.split(generated_sources, bsz)
    tgt_tokens_list = torch.split(collated_samples['net_input']['src_tokens'], bsz)
    target_list = [tensor.clone()[:, 1:] for tensor in tgt_tokens_list]
    prev_output_tokens_list = [tensor.clone()[:, :-1] for tensor in tgt_tokens_list]
    src_lang, src_lang_tokens, tgt_lang, tgt_lang_tokens = \
        collated_samples['net_input']['tgt_lang'], collated_samples['net_input']['tgt_lang_tokens'], \
        collated_samples['net_input']['src_lang'], collated_samples['net_input']['src_lang_tokens']
    for tensor in prev_output_tokens_list:
        tensor.masked_fill_(tensor==EOS_IDX, PAD_IDX)
        tensor[:, 0].fill_(EOS_IDX)
    id_list = torch.split(collated_samples['id'], bsz)
    batch_list = [{
        'id': id,
        'nsentences': src_tokens.shape[0],
        'ntokens': (target != PAD_IDX).sum().item(),
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': (src_tokens != PAD_IDX).sum(dim=1),
            'src_lang': src_lang,
            'src_lang_tokens': LANG_TOKEN_DIC[src_lang],
            'tgt_lang': tgt_lang,
            'tgt_lang_tokens': LANG_TOKEN_DIC[tgt_lang],
            'prev_output_tokens': prev_output_tokens
        },
        'target': target.contiguous(),
    } for src_tokens, target, id, prev_output_tokens in zip(src_tokens_list, target_list, id_list, prev_output_tokens_list)]
    return batch_list
    # return [
    #     {'id': id.item(), 'target': id_to_src[id.item()], 'source': hypos.cpu() if not isinstance(hypos, list) else hypos[0]['tokens'].cpu()}
    #     for id, hypos in zip(collated_samples['id'], generated_sources)
    # ]


class BacksummarizationDatasetV2(FairseqDataset):

    def __init__(
        self,
        tgt_dataset,
        src_dict,
        tgt_dict=None,
        backtranslation_fn=None,
        lang=None,
        max_generate_len=1024,
        cuda=True,
        accumulate_step=2,
        **kwargs
    ):
        self.tgt_dataset = tgt_dataset
        self.backtranslation_fn = backtranslation_fn
        self.lang = lang
        self.max_generate_len = max_generate_len
        self.cuda = cuda if torch.cuda.is_available() else False
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.accumulate_step = accumulate_step
        self.cur_accumulate = 0
        self.accumulate_sample = []

    def __getitem__(self, index):
        """
        Returns a single sample from *tgt_dataset*. Note that backtranslation is
        not applied in this step; use :func:`collater` instead to backtranslate
        a batch of samples.
        """
        return self.tgt_dataset[index]

    def __len__(self):
        return len(self.tgt_dataset)

    def set_backtranslation_fn(self, backtranslation_fn):
        self.backtranslation_fn = backtranslation_fn
    
    def collater(self, samples):
        # if samples[0].get('is_dummy', False):
        #     return samples
        # return self.tgt_dataset.collater(samples)
        return samples

    def reset(self):
        self.accumulate_sample = []
        self.cur_accumulate = 0

    def real_collater(self, samples):

        self.cur_accumulate += 1
        self.accumulate_sample += samples
        assert self.cur_accumulate <= self.accumulate_step
        if self.cur_accumulate == self.accumulate_step:
            all_samples = backtranslate_samples(
                samples=self.accumulate_sample,
                collate_fn=self.tgt_dataset.collater,
                generate_fn=(
                    lambda net_input: self.backtranslation_fn(net_input)
                ),
                accumulate_step=self.accumulate_step,
                lang=self.lang,
                max_generate_len=self.max_generate_len,
                cuda=self.cuda,
            )
            self.accumulate_sample = []
            self.cur_accumulate = 0
            
            return all_samples
        else:
            return None

    def num_tokens(self, index):
        """Just use the tgt dataset num_tokens"""
        return self.tgt_dataset.num_tokens(index)

    def ordered_indices(self):
        """Just use the tgt dataset ordered_indices"""
        return self.tgt_dataset.ordered_indices()

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used
        when filtering a dataset with ``--max-positions``.

        Note: we use *tgt_dataset* to approximate the length of the source
        sentence, since we do not know the actual length until after
        backtranslation.
        """
        tgt_size = self.tgt_dataset.size(index)[0]
        return (tgt_size, tgt_size)

    @property
    def supports_prefetch(self):
        return getattr(self.tgt_dataset, 'supports_prefetch', False)

    def prefetch(self, indices):
        return self.tgt_dataset.prefetch(indices)
