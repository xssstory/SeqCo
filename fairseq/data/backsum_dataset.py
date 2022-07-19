# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import math
import numpy as np
import copy

from fairseq import utils
from fairseq.data.backsum_pair_dataset import LANG_TOKEN_DIC
from . import FairseqDataset

PAD_IDX = 1
EOS_IDX = 2

def insert_sep_to_src(samples, sep_idx):
    new_samples = copy.deepcopy(samples)
    def insert(sample):
        dot_token = [4, 479]
        src  = sample['source']
        src_sentences = src.numpy()
        src_sentences = np.split(src_sentences, np.where(np.bitwise_or(src_sentences==dot_token[0], src_sentences==dot_token[1]))[0]+1)
        if len(src_sentences[-1]) == 1:
            if not src_sentences[-1].item() == EOS_IDX:
                raise TypeError('The last token mast be </s> !')
            else:
                del src_sentences[-1]
        else:
            assert src_sentences[-1][-1] == EOS_IDX
            src_sentences[-1] = src_sentences[-1][:-1]

        for idx, src_sentence in enumerate(src_sentences):
            src_sentences[idx] = np.append(src_sentence, sep_idx)

        src_sentences += [np.array([EOS_IDX], dtype=src_sentences[0].dtype)]
        out_tokens = torch.from_numpy(np.concatenate(src_sentences))
        sample['source'] = out_tokens
        return sample

    for idx, sample in enumerate(new_samples):
        new_samples[idx] = insert(sample)
    return new_samples

# @torch.no_grad()
def backtranslate_samples(samples, collate_fn, generate_fn, accumulate_step, cuda=True, insert_sep=False, sep_idx=None, need_pg=False):

    with torch.set_grad_enabled(need_pg):
        samples = utils.move_to_cpu(samples)
        if insert_sep and sep_idx:
            src_samples = insert_sep_to_src(samples, sep_idx)
            tgt_samples = samples
            collated_src_samples = collate_fn(src_samples)
            collated_src_samples = utils.move_to_cuda(collated_src_samples) if cuda else collated_src_samples
            collated_tgt_samples = collate_fn(tgt_samples)
            collated_tgt_samples = utils.move_to_cuda(collated_tgt_samples) if cuda else collated_tgt_samples
        else:
            collated_samples = collate_fn(samples)
            collated_samples = utils.move_to_cuda(collated_samples) if cuda else collated_samples
            collated_src_samples = collated_samples
            collated_tgt_samples = collated_samples
        generated_sources, action_log_probs = generate_fn(collated_src_samples)

    # id_to_src = {
    #     sample['id']: sample['source'] for sample in samples
    # }
    # bsz = len(generated_sources) // accumulate_step
    bsz = math.ceil(len(generated_sources) / accumulate_step)
    src_tokens_list = torch.split(generated_sources, bsz)
    tgt_tokens_list = torch.split(collated_tgt_samples['net_input']['src_tokens'], bsz)
    if collated_samples['target'] is not None:
        gold_src = torch.ones_like(collated_tgt_samples['target']).fill_(PAD_IDX)
        gold_src[:, 1:] = collated_tgt_samples['target'][:, :-1]
        gold_src.masked_fill_(gold_src==EOS_IDX, PAD_IDX)
        gold_src[:, 0].fill_(EOS_IDX)
        gold_src_tokens_list = torch.split(gold_src, bsz)
    else:
        gold_src_tokens_list = [None] * accumulate_step
    target_list = [tensor.clone()[:, 1:] for tensor in tgt_tokens_list]
    if target_list[0].size(1) == 0:
        target_list = [tensor.clone()[:, 1:] for tensor in src_tokens_list]

    prev_output_tokens_list = [tensor.clone()[:, :-1] for tensor in tgt_tokens_list]
    if prev_output_tokens_list[0].size(1) == 0:
        prev_output_tokens_list = [tensor.clone()[:, :-1] for tensor in src_tokens_list]

    src_lang, src_lang_tokens, tgt_lang, tgt_lang_tokens = \
        collated_tgt_samples['net_input']['tgt_lang'], collated_tgt_samples['net_input']['tgt_lang_tokens'], \
        collated_tgt_samples['net_input']['src_lang'], collated_tgt_samples['net_input']['src_lang_tokens']
    for tensor in prev_output_tokens_list:
        tensor.masked_fill_(tensor==EOS_IDX, PAD_IDX)
        tensor[:, 0].fill_(EOS_IDX)
    id_list = torch.split(collated_tgt_samples['id'], bsz)
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
        'gold_source': gold_src,
        'target': target.contiguous(),
    } for src_tokens, target, gold_src, id, prev_output_tokens in zip(src_tokens_list, target_list, gold_src_tokens_list, id_list, prev_output_tokens_list)]
    assert len(src_tokens_list) == len(target_list) == len(gold_src_tokens_list) == len(id_list) == len(prev_output_tokens_list)
    return batch_list, action_log_probs


class BacksummarizationDataset(FairseqDataset):
    """
    Sets up a backtranslation dataset which takes a tgt batch, generates
    a src using a tgt-src backtranslation function (*backtranslation_fn*),
    and returns the corresponding `{generated src, input tgt}` batch.

    Args:
        tgt_dataset (~fairseq.data.FairseqDataset): the dataset to be
            backtranslated. Only the source side of this dataset will be used.
            After backtranslation, the source sentences in this dataset will be
            returned as the targets.
        src_dict (~fairseq.data.Dictionary): the dictionary of backtranslated
            sentences.
        tgt_dict (~fairseq.data.Dictionary, optional): the dictionary of
            sentences to be backtranslated.
        backtranslation_fn (callable, optional): function to call to generate
            backtranslations. This is typically the `generate` method of a
            :class:`~fairseq.sequence_generator.SequenceGenerator` object.
            Pass in None when it is not available at initialization time, and
            use set_backtranslation_fn function to set it when available.
        output_collater (callable, optional): function to call on the
            backtranslated samples to create the final batch
            (default: ``tgt_dataset.collater``).
        cuda: use GPU for generation
    """

    def __init__(
        self,
        tgt_dataset,
        src_dict,
        tgt_dict=None,
        backtranslation_fn=None,
        insert_sep=False,
        cuda=True,
        accumulate_step=2,
        **kwargs
    ):
        self.tgt_dataset = tgt_dataset
        self.backtranslation_fn = backtranslation_fn
        self.insert_sep = insert_sep
        self.cuda = cuda if torch.cuda.is_available() else False
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.accumulate_step = accumulate_step
        self.cur_accumulate = 0
        self.accumulate_sample = []
        self.sep_idx = src_dict.index('<sep>')
        assert self.sep_idx != src_dict.unk()

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

    def real_collater(self, samples, need_pg=False):
        """Merge and backtranslate a list of samples to form a mini-batch.

        Using the samples from *tgt_dataset*, load a collated target sample to
        feed to the backtranslation model. Then take the backtranslation with
        the best score as the source and the original input as the target.

        Note: we expect *tgt_dataset* to provide a function `collater()` that
        will collate samples into the format expected by *backtranslation_fn*.
        After backtranslation, we will feed the new list of samples (i.e., the
        `(backtranslated source, original source)` pairs) to *output_collater*
        and return the result.

        Args:
            samples (List[dict]): samples to backtranslate and collate

        Returns:
            dict: a mini-batch with keys coming from *output_collater*
        """
        # if samples[0].get('is_dummy', False):
        #     return samples
        self.cur_accumulate += 1
        self.accumulate_sample += samples
        assert self.cur_accumulate <= self.accumulate_step
        if self.cur_accumulate == self.accumulate_step:
            all_samples, action_log_probs = backtranslate_samples(
                samples=self.accumulate_sample,
                collate_fn=self.tgt_dataset.collater,
                generate_fn=(
                    lambda net_input: self.backtranslation_fn(net_input)
                ),
                accumulate_step=self.accumulate_step,
                cuda=self.cuda,
                insert_sep=self.insert_sep,
                sep_idx=self.sep_idx,
                need_pg=need_pg,
            )
            self.accumulate_sample = []
            self.cur_accumulate = 0
            # assert len(all_samples) % self.accumulate_step == 0
            # bsz = len(all_samples) // self.accumulate_step
            # sample_list = [all_samples[i: i+bsz] for i in range(self.accumulate_step)]
            # return [utils.move_to_cuda(self.output_collater(samples)) if self.cuda else self.output_collater(samples) for samples in sample_list]
            return all_samples, action_log_probs
        else:
            return None, None

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
