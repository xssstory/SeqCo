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
import s2s_ft.s2s_loader as seq2seq_loader

SRC_TYPE_ID = 0
TGT_TYPE_ID = 1

def generate_instance(sample, max_source_len, max_target_len, d):

    input_ids = sample['source'].tolist()
    if input_ids[0] != d.bos():
        input_ids = [d.bos()] + input_ids
    if input_ids[-1] != d.eos():
        input_ids = input_ids + [d.eos()]
    if len(input_ids) > max_source_len:
        input_ids = input_ids[: max_source_len - 1] + [d.eos()]

    valid_src_len = len(input_ids)
    if len(input_ids) < max_source_len:
        input_ids += [d.pad()] * (max_source_len - len(input_ids))

    assert len(input_ids) == max_source_len

    max_len_in_batch = max_source_len + max_target_len
    segment_ids = [SRC_TYPE_ID] * max_source_len + [TGT_TYPE_ID] * max_target_len
    
    mask_qkv = None

    position_ids = []
    for i in range(valid_src_len):
        position_ids.append(i)
    for i in range(valid_src_len, max_source_len):
        position_ids.append(0)
    for i in range(max_source_len, max_len_in_batch):
        position_ids.append(i - max_source_len + valid_src_len)

    input_mask = torch.zeros(
        max_len_in_batch, max_len_in_batch, dtype=torch.long)

    input_mask[:, :valid_src_len].fill_(1)

    second_st, second_end = max_source_len, max_len_in_batch

    tril_matrix = torch.tril(torch.ones((max_len_in_batch, max_len_in_batch), dtype=torch.long))
    input_mask[second_st:second_end, second_st:second_end].copy_(tril_matrix[:second_end-second_st, :second_end-second_st])

    return (input_ids, segment_ids, position_ids, input_mask, mask_qkv)

# @torch.no_grad()
def backtranslate_samples(samples, collate_fn, generate_fn, accumulate_step, max_src_len, gen_tgt_len, cuda=True, d=None, need_pg=False):

    with torch.set_grad_enabled(need_pg):
        samples = utils.move_to_cpu(samples)
        collated_samples = collate_fn(samples)
        collated_samples = utils.move_to_cuda(collated_samples) if cuda else collated_samples

        instances = [generate_instance(sample, max_source_len=max_src_len, max_target_len=gen_tgt_len, d=d) for sample in samples]
        batch = seq2seq_loader.batch_list_to_batch_tensors(instances)
        batch = utils.move_to_cuda(batch) if cuda else batch
        input_ids, token_type_ids, position_ids, input_mask, mask_qkv = batch
        traces = generate_fn(input_ids, token_type_ids,
                    position_ids, input_mask, mask_qkv=mask_qkv)
        generated_target = traces['pred_seq'][:, :gen_tgt_len]
        generated_target = torch.cat([torch.ones([generated_target.shape[0], 1]).type_as(generated_target) 
        * d.bos(), generated_target], dim=1)
    
    bsz = math.ceil(len(generated_target) / accumulate_step)
    generated_list = torch.split(generated_target, bsz)
    src_tokens_list = torch.split(input_ids, bsz)

    if collated_samples['target'] is not None:
        gold_target = torch.ones_like(collated_samples['target']).fill_(d.pad())
        gold_target[:, 1:] = collated_samples['target'][:, :-1]
        gold_target[:, -1].masked_fill_((gold_target[:, -1]!=d.eos()) & (gold_target[:, -1]!=d.pad()), d.eos())

        gold_target[:, 0].fill_(d.bos())
        gold_tgt_tokens_list = torch.split(gold_target, bsz)
    else:
        gold_tgt_tokens_list = [None] * accumulate_step

    id_list = torch.split(collated_samples['id'], bsz)
    batch_list = [{
        'id': id,
        'nsentences': src_tokens.shape[0],
        'ntokens': (src_tokens != d.pad()).sum().item(),
        'src_tokens': src_tokens,
        'gold_target': gold_tgt,
        "generated_target": gene_tgt
    } for src_tokens, gold_tgt, gene_tgt, id in zip(src_tokens_list, gold_tgt_tokens_list, generated_list, id_list)]
    assert len(src_tokens_list) == len(gold_tgt_tokens_list) == len(generated_list) == len(id_list) == len(id_list)
    return batch_list, None


class UnilmBackDataset(FairseqDataset):
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
        max_src_len,
        gen_tgt_len,
        tgt_dict=None,
        backtranslation_fn=None,
        cuda=True,
        accumulate_step=2,
        **kwargs
    ):
        self.tgt_dataset = tgt_dataset
        self.backtranslation_fn = backtranslation_fn
        self.cuda = cuda if torch.cuda.is_available() else False
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.accumulate_step = accumulate_step
        self.cur_accumulate = 0
        self.accumulate_sample = []
        self.max_src_len = max_src_len
        self.gen_tgt_len = gen_tgt_len

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
                # generate_fn=(
                #     lambda net_input: self.backtranslation_fn(net_input)
                # ),
                generate_fn=self.backtranslation_fn,
                accumulate_step=self.accumulate_step,
                max_src_len=self.max_src_len,
                gen_tgt_len=self.gen_tgt_len,
                cuda=self.cuda,
                d=self.tgt_dict,
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
