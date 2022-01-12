# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from collections import OrderedDict
import os
import copy

from fairseq import options, utils
from fairseq.data import (
    IndexedCachedDataset,
    IndexedDataset,
    IndexedRawTextDataset,
    # LanguagePairDataset,
    # NoisingDataset,
    RoundRobinZipDatasets,
    Dictionary,
    data_utils,
    TransformEosLangPairDataset,
    AppendTokenDataset,
    TruncateDataset,
    StripTokenDataset,
)
from fairseq.data.bart_dataset import BartDataset
from fairseq.data.sent_noising import NoisingDataset
from fairseq.data.extract import ExtractDataset
from fairseq.data.upsampling import UpsamplingDataset
from fairseq.data.backsum_dataset import BacksummarizationDataset
from fairseq.data.backsum_pair_dataset import BacksumPairDataset
from fairseq.models import FairseqMultiModel
from fairseq.sequence_generator import SequenceGenerator


from .multilingual_translation import MultilingualTranslationTask
from .fairseq_task import FairseqTask
from . import register_task


def _get_bt_dataset_key(lang):
    return "bt:" + lang


def _get_denoising_dataset_key(lang):
    return "denoising:" + lang

def _get_bart_dataset_key(lang):
    return "bart:" + lang


# ported from UnsupervisedMT
def parse_lambda_config(x):
    """
    Parse the configuration of lambda coefficient (for scheduling).
    x = "3"                  # lambda will be a constant equal to x
    x = "0:1,1000:0"         # lambda will start from 1 and linearly decrease
                             # to 0 during the first 1000 iterations
    x = "0:0,1000:0,2000:1"  # lambda will be equal to 0 for the first 1000
                             # iterations, then will linearly increase to 1 until iteration 2000
    """
    if isinstance(x, float):
        return x, None
    split = x.split(',')
    if len(split) == 1:
        return float(x), None
    else:
        split = [s.split(':') for s in split]
        assert all(len(s) == 2 for s in split)
        assert all(k.isdigit() for k, _ in split)
        assert all(int(split[i][0]) < int(split[i + 1][0]) for i in range(len(split) - 1))
        return float(split[0][1]), [(int(k), float(v)) for k, v in split]


def add_special_token_to_dict(d, tok):
    if tok in d.indices:
        return d.indices[tok]
    else:
        for i in range(len(d.symbols)):
            if d.symbols[i].startswith('[unused') or d.symbols[i].startswith('madeupword'):
                orig_tok = d.symbols[i]
                d.symbols[i] = tok
                del d.indices[orig_tok]
                d.indices[tok] = i
                print('add {} to dict'.format(tok))
                return i

        print('No space for token "{}"'.format(tok))
        return None

def truncate_dataset(dataset, eos, max_position):
    dataset = AppendTokenDataset(
        TruncateDataset(
            StripTokenDataset(dataset, eos),
            max_position - 1,
        ),
        eos,
    )
    return dataset


# def generate(self, src_enc, src_len, tgt_lang_id, max_len=200, sample_temperature=None):
def generate(model, sample, d, bos_token=None, max_len=200, sample_temperature=1):
    """
    Decode a sentence given initial start.
    `x`:
        - LongTensor(bs, slen)
            <EOS> W1 W2 W3 <EOS> <PAD>
            <EOS> W1 W2 W3   W4  <EOS>
    `lengths`:
        - LongTensor(bs) [5, 6]
    `positions`:
        - False, for regular "arange" positions (LM)
        - True, to reset positions from the new generation (MT)
    `langs`:
        - must be None if the model only supports one language
        - lang_id if only one language is involved (LM)
        - (lang_id1, lang_id2) if two languages are involved (MT)
    """
    model.eval()
    encoder_input = {
        k: v for k, v in sample['net_input'].items()
        if k != 'prev_output_tokens'
    }
    src_tokens = encoder_input['src_tokens']
    src_len = (src_tokens.ne(d.eos()) & src_tokens.ne(d.pad())).long().sum(dim=1)
    with torch.no_grad():
        encoder_out = model.encoder(lang_tokens=encoder_input['src_lang_tokens'], **encoder_input)
    # input batch
    bs = len(src_len)

    # generated sentences
    generated = src_len.new(bs, max_len)  # upcoming output
    generated.fill_(d.pad())       # fill upcoming ouput with <PAD>
    bos_token = bos_token or d.eos()
    generated[:, 0].fill_(bos_token)    # we use <EOS> for <BOS> everywhere

    action_log_probs = []

    # current position / max lengths / length of generated sentences / unfinished sentences
    cur_len = 1
    incremental_state = {}
    gen_len = src_len.clone().fill_(1)
    unfinished_sents = src_len.clone().fill_(1)

    # cache compute states
    cache = {'slen': 0}

    while cur_len < max_len:

        decoder_out = list(model.forward_decoder(generated[:, :cur_len], encoder_out=encoder_out, incremental_state=incremental_state, lang_tokens=encoder_input.get('tgt_lang_tokens')))
        # decoder_out[0] = decoder_out[0][:, -1:, :]
        # scores = model.get_normalized_probs(decoder_out, log_probs=True)
        scores = F.log_softmax(decoder_out[0], dim=-1)
        assert decoder_out[0].size(1) == 1 and scores.size(1) == 1
        # scores = scores[:, -1, :]
        scores = scores.squeeze(1)
        # next_words = torch.topk(scores, 1)[1].squeeze(1)
        # select next words: sample or greedy
        if sample_temperature is None:
            next_words = torch.topk(scores, 1)[1].squeeze(1)
        else:
            next_words = torch.multinomial(torch.functional.F.softmax(scores / sample_temperature, dim=1), 1).squeeze(1)
        assert next_words.size() == (bs,)

        action_log_prob = torch.gather(scores, 1, next_words.unsqueeze(1)).squeeze(1)

        # update generations / lengths / finished sentences / current length
        generated[:, cur_len] = next_words * unfinished_sents + d.pad() * (1 - unfinished_sents)
        # action_log_probs[:, cur_len] = action_log_prob * unfinished_sents.to(action_log_prob.dtype)
        action_log_probs.append(action_log_prob * unfinished_sents.clone().to(action_log_prob.dtype))

        gen_len.add_(unfinished_sents)
        unfinished_sents.mul_(next_words.ne(d.eos()).long())
        cur_len = cur_len + 1

        # stop when there is a </s> in each sentence, or if we exceed the maximul length
        if unfinished_sents.max() == 0:
            break

    # add <EOS> to unfinished sentences
    if cur_len == max_len:
        generated[:, -1].masked_fill_(unfinished_sents.bool(), d.eos())

    # sanity check
    if bos_token == d.eos():
        (generated == d.eos()).sum() == bs * 2
    else:
        assert (generated == d.eos()).sum() == bs

    return generated, torch.stack(action_log_probs, dim=1)



@register_task('finetune_summarization')
class FinetuneSummarizationTask(FairseqTask):

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off

        parser.add_argument('data', metavar='DIR', help='path to data directory')
        # parser.add_argument('--lang-pairs', default=None, metavar='PAIRS',
        #                     help='comma-separated list of language pairs (in training order): en-de,en-fr,de-fr')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language (only needed for inference)')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language (only needed for inference)')
        parser.add_argument('--lazy-load', action='store_true',
                            help='load the dataset lazily')
        parser.add_argument('--raw-text', default=False, action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left (default: True)')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left (default: False)')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        parser.add_argument('--change-langtok', default=False, action='store_true')
        parser.add_argument('--accumulate-trans', default=None, type=int)
        parser.add_argument('--two-side', default=False, action='store_true')
        parser.add_argument('--symmetrical', default='False', type=eval)
        parser.add_argument('--cross-byol', default='True', type=eval)
        # parser.add_argument('--encoder-langtok', default=None, type=str, choices=['src', 'tgt'],
        #                     metavar='SRCTGT',
        #                     help='replace beginning-of-sentence in source sentence with source or target '
        #                          'language token. (src/tgt)')
        # parser.add_argument('--decoder-langtok', action='store_true',
        #                     help='replace beginning-of-sentence in target sentence with target language token')

        parser.add_argument('--lambda-parallel-config', default="0.0", type=str, metavar='CONFIG',
                            help='cross-entropy reconstruction coefficient (parallel data). '
                                 'use fixed weight during training if set to floating point number. '
                                 'use piecewise linear function over number of updates to schedule the '
                                 'weight with the format: w0:step0,w1:step1,...')
        parser.add_argument('--truncate-source-positions', default=1024, type=int)
        parser.add_argument('--insert-sep', default='False', type=str)
        parser.add_argument('--decoder-byol', default='0', type=str)
        parser.add_argument('--momentum-contrast-loss-ratio', default="0", type=str)
        parser.add_argument('--momentum-contrast-capcity', default=10000, type=int)
        parser.add_argument('--momentum-contrast-beta', default="0.999", type=str)
        parser.add_argument('--momentum-contrast-t', default=1, type=float)
        parser.add_argument('--byol-ratio', default="0", type=str)
        parser.add_argument('--parallel-byol-ratio', default='0', type=str)
        parser.add_argument('--gen-sum-len', default=140, type=int)
        parser.add_argument('--gold-gen-byol', default='0', type=str)
        parser.add_argument('--pg-ratio', default="0", type=str)
        parser.add_argument('--lambda-bart-pretrain-config', default="0.0", type=str, metavar='CONFIG')
        parser.add_argument('--lambda-denoising-config', default="1.0", type=str, metavar='CONFIG',
                            help='Cross-entropy reconstruction coefficient (denoising autoencoding)'
                                 'use fixed weight during training if set to floating point number. '
                                 'use piecewise linear function over number of updates to schedule the '
                                 'weight with the format: w0:step0,w1:step1,...')
        parser.add_argument('--lambda-otf-bt-config', default="1.0", type=str, metavar='CONFIG',
                            help='cross-entropy reconstruction coefficient (on-the-fly back-translation parallel data)'
                                 'use fixed weight during training if set to floating point number. '
                                 'use piecewise linear function over number of updates to schedule the '
                                 'weight with the format: w0:step0,w1:step1,...')
        parser.add_argument('--article-downsampling', default='True', type=str)
        parser.add_argument('--summary-upsampling', default='True', type=str)
        parser.add_argument('--bt-beam-size', default=1, type=int, metavar='N',
                            help='beam size used in beam search of online back-translation')
        parser.add_argument('--max-word-shuffle-distance', default=3.0, type=float, metavar='N',
                            help='maximum word shuffle distance for denoising autoencoding data generation')
        parser.add_argument('--word-dropout-prob', default=0.1, type=float, metavar='N',
                            help='word dropout probability for denoising autoencoding data generation')
        parser.add_argument('--word-blanking-prob', default=0.2, type=float, metavar='N',
                            help='word blanking probability for denoising autoencoding data generation')
        parser.add_argument('--bart-mask-full-sent', default='False', type=str)
        parser.add_argument('--init-from-pretrained-doc-model', action='store_true')
        parser.add_argument('--pretrained-doc-model-path', type=str, default='')
        # fmt: on

    def __init__(self, args, d, training):
        super().__init__(args)

        self.dict = d
        self.training = training
        if training:
            self.langs = args.langs
            assert len(self.langs) == 2, 'only two domain article supported now !'
        self.eval_lang_pairs = ['{}-{}'.format(args.source_lang, args.target_lang)]

        self.model_lang_pairs = ["{}-{}".format(args.source_lang, args.target_lang)]
        if args.two_side:
            self.model_lang_pairs.append("{}-{}".format(args.target_lang, args.source_lang))
        else:
            assert eval(args.lambda_otf_bt_config) == 0, "--two-side is needed to use bt config"
        for lang in self.langs:
            add_special_token_to_dict(self.dict, '[{}_bos]'.format(lang))
        
        add_special_token_to_dict(self.dict, '<sep>')

        self.lambda_otf_bt, self.lambda_otf_bt_steps = parse_lambda_config(args.lambda_otf_bt_config)
        self.lambda_denoising, self.lambda_denoising_steps = parse_lambda_config(args.lambda_denoising_config)
        self.lambda_bart_pretrain, self.lambda_bart_pretrain_steps = parse_lambda_config(args.lambda_bart_pretrain_config)
        self.lambda_parallel, self.lambda_parallel_steps = parse_lambda_config(args.lambda_parallel_config)
        self.byol_ratio, self.byol_steps = parse_lambda_config(args.byol_ratio)
        self.decoder_byol_ratio, self.decoder_byol_steps = parse_lambda_config(args.decoder_byol)
        self.parallel_byol_ratio, self.parallel_byol_steps = parse_lambda_config(args.parallel_byol_ratio)
        self.gold_gen_byol_ratio, self.gold_gen_steps = parse_lambda_config(args.gold_gen_byol)
        self.momentum_contrast_loss_ratio, self.moco_steps = parse_lambda_config(args.momentum_contrast_loss_ratio)
        self.momentum_contrast_beta, self.momentum_beta_steps = parse_lambda_config(args.momentum_contrast_beta)
        self.pg_ratio, self.pg_steps = parse_lambda_config(args.pg_ratio)
        self.backtranslate_datasets = {}
        self.backtranslators = {}

        self.lang_token_dic = {
            'article': 'article',
            'summary': 'summary',
        }

    @classmethod
    def setup_task(cls, args, **kwargs):
        d, training = cls.prepare(args, **kwargs)
        return cls(args, d, training)

    @classmethod
    def prepare(cls, args, **kargs):
        args.left_pad_source = False
        args.left_pad_target = False
        if not isinstance(args.article_downsampling, bool):
            args.article_downsampling = eval(args.article_downsampling)
        if not isinstance(args.summary_upsampling, bool):
            args.summary_upsampling = eval(args.summary_upsampling)
        if not isinstance(args.insert_sep, bool):
            args.insert_sep = eval(args.insert_sep)
        if not isinstance(args.bart_mask_full_sent, bool):
            args.bart_mask_full_sent = eval(args.bart_mask_full_sent)
        if args.accumulate_trans is None:
            args.accumulate_trans = min(args.update_freq)
        else:
            args.accumulate_trans = min(args.accumulate_trans, min(args.update_freq))
        if getattr(args, 'raw_text', False):
            utils.deprecation_warning('--raw-text is deprecated, please use --dataset-impl=raw')
            args.dataset_impl = 'raw'
        elif getattr(args, 'lazy_load', False):
            utils.deprecation_warning('--lazy-load is deprecated, please use --dataset-impl=lazy')
            args.dataset_impl = 'lazy'

        args.langs = [args.source_lang, args.target_lang]
        
        sorted_langs = sorted(list({x for lang_pair in args.langs for x in lang_pair.split('-')}))
        # if args.source_lang is not None or args.target_lang is not None:
        #     training = False
        # else:
        #     training = True
        training = True
        
        paths = args.data.split(':')
        for lang in sorted_langs:
            dict_path = os.path.join(paths[0], 'dict.{}.txt'.format(lang))
            if os.path.exists(dict_path):
                d = Dictionary.load(dict_path)
                break
            else:
                continue
        if not 'd' in locals():
            raise FileNotFoundError('Dict file is needed !')
        elif len(d) != 50265:
            assert len(d) == 50264
            d.add_symbol('<mask>')

        print('| [{}] : {} types'.format('Dictionary', len(d)))
        return d, training

    def load_dataset(self, split, epoch=0, **kwargs):
        """Load a dataset split."""

        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]

        def split_exists(split, src, tgt, lang):
            if src is not None:
                filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
            else:
                filename = os.path.join(data_path, '{}.{}-None.{}'.format(split, src, tgt))
            if self.args.raw_text and IndexedRawTextDataset.exists(filename):
                return True
            elif not self.args.raw_text and IndexedDataset.exists(filename):
                return True
            return False

        def indexed_dataset(path, dictionary):
            dataset_impl = None
            if getattr(self.args, 'raw_text', False):
                dataset_impl = 'raw'
            elif getattr(self.args, 'lazy_load', False):
                dataset_impl = 'lazy'
            return truncate_dataset(data_utils.load_indexed_dataset(path, dictionary, dataset_impl), eos=self.dict.eos(), max_position=self.args.truncate_source_positions)

        def get_filename(split, lang1, lang2):
            if split_exists(split, lang1, None, lang1):
                filename = os.path.join(data_path, '{}.{}-None.{}'.format(split, lang1, lang1))
            elif split_exists(split, lang1, lang2, lang1):
                filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, lang1, lang2, lang1))
            elif split_exists(split, lang2, lang1, lang1):
                filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, lang2, lang1, lang1))
            else:
                raise FileNotFoundError('Dataset not found: at least one of {}.{}-None.{}, {}.{}-{}.{} and {}.{}-{}.{} exists'.format(
            split, lang1, lang1, split, lang1, lang2, lang1, split, lang2, lang1, lang1))
            return filename



        # load parallel datasets
        src_datasets, tgt_datasets = {}, {}
        if not split.startswith('train') or self.lambda_parallel > 0.0:
            for lang_pair in self.model_lang_pairs:
                src, tgt = lang_pair.split('-')
                if not split.startswith('train') and src != self.args.source_lang:
                    continue
                if split_exists(split, src, tgt, src):
                    prefix = os.path.join(data_path, '{}.{}-{}.'.format(split, src, tgt))
                elif split_exists(split, tgt, src, src):
                    prefix = os.path.join(data_path, '{}.{}-{}.'.format(split, tgt, src))
                else:
                    raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))
                src_datasets[lang_pair] = indexed_dataset(prefix + src, self.dict)
                tgt_datasets[lang_pair] = indexed_dataset(prefix + tgt, self.dict)
                print('| parallel-{} {} {} examples'.format(data_path, split, len(src_datasets[lang_pair])))

        # back translation datasets
        backtranslate_datasets = {}
        if (self.lambda_otf_bt > 0.0 or self.lambda_otf_bt_steps is not None 
        or self.momentum_contrast_loss_ratio > 0.0 or self.moco_steps is not None
        or self.byol_ratio > 0.0 or self.byol_steps is not None
        or self.gold_gen_byol_ratio > 0.0 or self.gold_gen_steps is not None
        or self.decoder_byol_ratio > 0.0 or self.decoder_byol_steps is not None) and split.startswith("train"):
            lang_set = set(self.langs)
            for lang in self.langs:
                tgt = lang
                src = (lang_set - {tgt}).pop()
                if "{}-{}".format(tgt, src) not in self.model_lang_pairs:
                    continue
                # if not split_exists(split, tgt, None, tgt):
                #     raise FileNotFoundError('Dataset not found: backtranslation {} ({})'.format(split, data_path))
                # filename = os.path.join(data_path, '{}.{}-None.{}'.format(split, tgt, tgt))
                filename = get_filename(split, tgt, src)
                dataset = indexed_dataset(filename, self.dict)
                if (self.gold_gen_byol_ratio > 0.0 or self.gold_gen_steps is not None
                or self.decoder_byol_ratio > 0.0 or self.decoder_byol_steps is not None):
                    filename_src = get_filename(split, src, tgt)
                    dataset_src = indexed_dataset(filename_src, self.dict)
                    src_sizes = dataset_src.sizes
                else:
                    dataset_src = None
                    src_sizes = None

                lang_pair_dataset_tgt = BacksumPairDataset(
                    dataset,
                    dataset.sizes,
                    self.dict,
                    dataset_src,
                    src_sizes,
                    self.dict,
                    left_pad_source=self.args.left_pad_source,
                    left_pad_target=self.args.left_pad_target,
                    src_lang=tgt,
                    tgt_lang=src,
                    append_bos=True,
                )
                # lang_pair_dataset = BacksumPairDataset(
                #     dataset,
                #     dataset.sizes,
                #     src_dict=self.dict,
                #     tgt=dataset,
                #     tgt_sizes=dataset.sizes,
                #     tgt_dict=self.dict,
                #     left_pad_source=self.args.left_pad_source,
                #     left_pad_target=self.args.left_pad_target,
                #     src_lang=src,
                #     tgt_lang=tgt,
                #     append_bos=True,
                # )
                backtranslate_datasets[lang] = BacksummarizationDataset(
                    tgt_dataset = lang_pair_dataset_tgt,
                    backtranslation_fn=self.backtranslators[lang],
                    src_dict=self.dict, tgt_dict=self.dict,
                    insert_sep=(self.args.insert_sep and lang == self.lang_token_dic['summary']),
                    accumulate_step=self.args.accumulate_trans,
                )
                print('| backtranslate-{}: {} {} {} examples'.format(
                    tgt, data_path, split, len(backtranslate_datasets[lang]),
                ))
                self.backtranslate_datasets[lang] = backtranslate_datasets[lang]

        # denoising autoencoder
        noising_datasets = {}
        if (self.lambda_denoising > 0.0 or self.lambda_denoising_steps is not None) and split.startswith("train"):
            lang_set = set(self.langs)
            for lang in self.langs:
                # _, tgt = lang_pair.split('-')
                tgt = lang
                another_lang = (lang_set - {tgt}).pop()
                if "{}-{}".format(another_lang, lang) not in self.model_lang_pairs:
                    continue
                # if not split_exists(split, tgt, None, tgt):
                #     continue
                # filename = os.path.join(data_path, '{}.{}-None.{}'.format(split, tgt, tgt))
                filename = get_filename(split, tgt, another_lang)

                tgt_dataset1 = indexed_dataset(filename, self.dict)
                if lang == self.lang_token_dic['article'] and self.args.article_downsampling:
                    tgt_dataset1 = ExtractDataset(
                        tgt_dataset1,
                        src_dict=self.dict,
                        seed=self.args.seed,
                        insert_sep=self.args.insert_sep,
                        # extract_num=3,
                    )
                if lang == self.lang_token_dic['summary'] and self.args.summary_upsampling:
                    filename2 = get_filename(split, another_lang, tgt)
                    candidate_dataset = indexed_dataset(filename2, self.dict)
                    tgt_dataset1 = UpsamplingDataset(
                        tgt_dataset1,
                        candidate_dataset, 
                        self.dict,
                        self.args.seed
                    )

                tgt_dataset2 = indexed_dataset(filename, self.dict)
                noising_dataset = NoisingDataset(
                    tgt_dataset1,
                    self.dict,
                    seed=self.args.seed,
                    max_word_shuffle_distance=self.args.max_word_shuffle_distance,
                    word_dropout_prob=self.args.word_dropout_prob,
                    word_blanking_prob=self.args.word_blanking_prob,
                )
                noising_datasets[lang] = self.alter_dataset_langtok(
                    BacksumPairDataset(
                        noising_dataset,
                        tgt_dataset1.sizes,
                        self.dict,
                        tgt_dataset2,
                        tgt_dataset2.sizes,
                        self.dict,
                        left_pad_source=self.args.left_pad_source,
                        left_pad_target=self.args.left_pad_target,
                        src_lang=another_lang,
                        tgt_lang=lang,
                        append_bos=True,
                    ),
                    src_eos=self.dict.eos(),
                    src_lang=tgt,
                    tgt_eos=self.dict.eos(),
                    tgt_lang=tgt,
                )
                print('| denoising-{}: {} {} {} examples'.format(
                    tgt, data_path, split, len(noising_datasets[lang]),
                ))
        #bart-pretrain
        bart_datasets = {}
        if (self.lambda_bart_pretrain > 0.0 or self.lambda_bart_pretrain_steps is not None) and split.startswith("train"):
            lang_set = set(self.langs)
            for lang in self.langs:
                tgt = lang
                another_lang = (lang_set - {tgt}).pop()
                if "{}-{}".format(another_lang, lang) not in self.model_lang_pairs:
                    continue
                # if not split_exists(split, tgt, None, tgt):
                #     continue
                # filename = os.path.join(data_path, '{}.{}-None.{}'.format(split, tgt, tgt))
                filename = get_filename(split, tgt, another_lang)
  
                tgt_dataset1 = indexed_dataset(filename, self.dict)
                tgt_dataset1 = BartDataset(
                    tgt_dataset1,
                    self.dict,
                    seed=self.args.seed,
                    mask_full_sent=self.args.bart_mask_full_sent,
                )

                tgt_dataset2 = indexed_dataset(filename, self.dict)
                bart_datasets[lang] = self.alter_dataset_langtok(
                    BacksumPairDataset(
                        tgt_dataset1,
                        tgt_dataset1.sizes,
                        self.dict,
                        tgt_dataset2,
                        tgt_dataset2.sizes,
                        self.dict,
                        left_pad_source=self.args.left_pad_source,
                        left_pad_target=self.args.left_pad_target,
                        src_lang=another_lang,
                        tgt_lang=lang,
                        append_bos=True,
                    ),
                    src_eos=self.dict.eos(),
                    src_lang=tgt,
                    tgt_eos=self.dict.eos(),
                    tgt_lang=tgt,
                )
                print('| bart_pretrain-{}: {} {} {} examples'.format(
                    tgt, data_path, split, len(bart_datasets[lang]),
                ))

        def language_pair_dataset(lang_pair):
            src, tgt = lang_pair.split('-')
            src_dataset, tgt_dataset = src_datasets[lang_pair], tgt_datasets[lang_pair]
            return self.alter_dataset_langtok(
                BacksumPairDataset(
                    src_dataset, src_dataset.sizes, self.dict,
                    tgt_dataset, tgt_dataset.sizes, self.dict,
                    left_pad_source=self.args.left_pad_source,
                    left_pad_target=self.args.left_pad_target,
                    max_source_positions=self.args.max_source_positions,
                    max_target_positions=self.args.max_target_positions,
                    src_lang=self.lang_token_dic[src],
                    tgt_lang=self.lang_token_dic[tgt],
                    append_bos=True,
                ),
                self.dict.eos(),
                src,
                self.dict.eos(),
                tgt,
            )

        self.datasets[split] = RoundRobinZipDatasets(
            OrderedDict([
                (lang_pair, language_pair_dataset(lang_pair))
                for lang_pair in self.model_lang_pairs if (split.startswith('train') or lang_pair == "%s-%s" % (self.args.source_lang, self.args.target_lang))
            ] + [
                (_get_bt_dataset_key(lang_pair), dataset)
                for lang_pair, dataset in backtranslate_datasets.items()
            ] + [
                (_get_denoising_dataset_key(lang_pair), dataset)
                for lang_pair, dataset in noising_datasets.items()
            ] + [
                (_get_bart_dataset_key(lang_pair), dataset)
                for lang_pair, dataset in bart_datasets.items()
            ]),
            eval_key=None if self.training else "%s-%s" % (self.args.source_lang, self.args.target_lang),
        )

    def build_model(self, args):
        from fairseq import models
        model = models.build_model(args, self)
        # if not isinstance(model, FairseqMultiModel):
        #     raise ValueError('SemisupervisedTranslationTask requires a FairseqMultiModel architecture')

        # create SequenceGenerator for each model that has backtranslation dependency on it
        self.sequence_generators = {}
        lang_set = set(self.langs)
        if (self.lambda_otf_bt > 0.0 or self.lambda_otf_bt_steps is not None 
        or self.momentum_contrast_loss_ratio > 0.0 or self.moco_steps is not None
        or self.byol_ratio > 0.0 or self.byol_steps is not None
        or self.gold_gen_byol_ratio or self.gold_gen_steps is not None
        or self.decoder_byol_ratio > 0.0 or self.decoder_byol_steps is not None) and self.training:
            for lang in self.langs:
  
                key, tgt = lang, lang
                src = (lang_set - {tgt}).pop()
                # decoder_lang_tok_idx = self.get_decoder_langtok(src)
                decoder_lang_tok_idx = self.dict.eos()
                min_len, max_len_a, max_len_b = self.generator_para[src]
                model_key = '{}-{}'.format(tgt, src)
                if model_key not in self.model_lang_pairs:
                    continue
                # self.sequence_generators[key] = SequenceGenerator(
                #     tgt_dict=self.dict,
                #     beam_size=args.bt_beam_size,
                #     min_len=min_len,
                #     max_len_a=max_len_a,
                #     max_len_b=max_len_b,
                # )

                # def backtranslate_fn(
                #     sample, model=model.models[model_key],
                #     bos_token=decoder_lang_tok_idx,
                #     sequence_generator=self.sequence_generators[key],
                # ):
                #     return sequence_generator.generate(
                #         [model],
                #         sample,
                #         bos_token=bos_token,
                #     )
                def backtranslate_fn(
                    sample,
                    model=model.models[model_key],
                    bos_token=decoder_lang_tok_idx,
                    max_len=max_len_b,
                ):
                    return generate(model=model, sample=sample, bos_token=bos_token, max_len=max_len, d=self.dict)
                self.backtranslators[lang] = backtranslate_fn

        return model

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        model.train()
        agg_loss, agg_sample_size, agg_logging_output = 0., 0., {}

        def forward_backward(model, samples, logging_output_key, weight):
            nonlocal agg_loss, agg_sample_size, agg_logging_output
            if samples is None or len(samples) == 0:
                return
            loss, sample_size, logging_output = criterion(model, samples)
            if ignore_grad:
                loss *= 0
            else:
                loss *= weight
            optimizer.backward(loss)
            agg_loss += loss.detach().item()
            # TODO make summing of the sample sizes configurable
            agg_sample_size += sample_size

            if logging_output_key not in agg_logging_output:
                agg_logging_output[logging_output_key] = logging_output
            else:
                agg_logging_output[logging_output_key] = {
                    k: agg_logging_output[logging_output_key][k] + logging_output[k] for k in logging_output.keys()
                }
        
        def decoder_byol_forward_backward(model, samples, logging_output_key, weight):

            def forward_single(input_q, input_k):
                key = list(model.models.keys())[0]
                encoder_out = model.models[key].encoder(input_q['src_tokens'], None, lang_tokens=input_q['src_lang_tokens'])
                decoder_out = model.models[key].decoder(
                        input_q['prev_output_tokens'], encoder_out, lang_tokens=input_q['tgt_lang_tokens'], features_only=True,
                    )
                q = decoder_out[0].transpose(0, 1)

                q = model.encoder_extra_fc(q)
                q = torch.nn.functional.normalize(q, dim=-1)
                with torch.no_grad():

                    encoder_out = model.momentum_encoder(input_k['src_tokens'], None, lang_tokens=input_k['src_lang_tokens'])
                    decoder_out = model.momentum_decoder(
                            input_k['prev_output_tokens'], encoder_out, lang_tokens=input_k['tgt_lang_tokens'], features_only=True,
                        )
                    k = decoder_out[0].transpose(0, 1)
                    k = model.momentum_encoder_extra_fc(k)
                    k = torch.nn.functional.normalize(k, dim=-1)
                n_tokens, bsz, embed_size = q.shape
                if self.args.cross_byol:
                    q = model.byol_cross_attention(q, k)
                    loss = (1 - torch.cosine_similarity(q.transpose(0, 1), k.transpose(0, 1), dim=-1)).mean()
                    sim = torch.cosine_similarity(q.transpose(0, 1), k.transpose(0, 1), dim=-1).detach()
                else:
                    q = model.encoder_extra_fc2(q[0])
                    k = k[0]
                    q = torch.nn.functional.normalize(q, dim=1)
                    loss = (1 - torch.cosine_similarity(q, k)).mean()
                    sim = torch.cosine_similarity(q, k).detach()

                sample_size = bsz
                return loss, n_tokens, bsz, sample_size, sim

            nonlocal agg_loss, agg_sample_size, agg_logging_output
            if samples is None or len(samples) == 0:
                return
            net_input = samples['net_input']

            gen_input = {}
            gold_input = {}

            gen_input['src_tokens'] = net_input['prev_output_tokens']
            gen_input['prev_output_tokens'] =  net_input['src_tokens']
            gen_input['src_lang_tokens'] = net_input['tgt_lang_tokens']
            gen_input['tgt_lang_tokens'] = net_input['src_lang_tokens']
            
            gold_input['src_tokens'] = net_input['prev_output_tokens']
            gold_input['prev_output_tokens'] = samples['gold_source']
            gold_input['src_lang_tokens'] = net_input['tgt_lang_tokens']
            gold_input['tgt_lang_tokens'] = net_input['src_lang_tokens']

            loss, n_tokens, bsz, sample_size, sim = forward_single(gold_input, gen_input)
            if self.args.symmetrical:
                loss2, n_tokens2, bsz2, sample_size2, _ = forward_single(gen_input, gold_input)
                loss = loss + loss2
                n_tokens += n_tokens2
                bsz += bsz2
                sample_size += sample_size2

            logging_output = {
                'loss': utils.item(loss.data),
                'ntokens': n_tokens,
                'nsentences': bsz,
                'sample_size': sample_size,
            }
            if ignore_grad:
                loss *= 0
            else:
                loss *= weight
            optimizer.backward(loss)

            agg_loss += loss.detach().item()
            # TODO make summing of the sample sizes configurable
            agg_sample_size += sample_size
            if logging_output_key not in agg_logging_output:
                agg_logging_output[logging_output_key] = logging_output
            else:
                agg_logging_output[logging_output_key] = {
                    k: agg_logging_output[logging_output_key][k] + logging_output[k] for k in logging_output.keys()
                }
            return sim

        def byol_forward_backward(model, samples, logging_output_key, weight, gold_gen=False):

            def forward_single(input_q, input_k):
                key = list(model.models.keys())[0]
                q = model.models[key].encoder(**input_q).encoder_out
                q = model.encoder_extra_fc(q)
                q = torch.nn.functional.normalize(q, dim=-1)
                with torch.no_grad():
                    k = model.momentum_encoder(**input_k).encoder_out.detach()
                    k = model.momentum_encoder_extra_fc(k)
                    k = torch.nn.functional.normalize(k, dim=-1)
                n_tokens, bsz, embed_size = q.shape
                if self.args.cross_byol:
                    q = model.byol_cross_attention(q, k)
                    loss = (1 - torch.cosine_similarity(q.transpose(0, 1), k.transpose(0, 1), dim=-1)).mean()
                    sim = torch.cosine_similarity(q.transpose(0, 1), k.transpose(0, 1), dim=-1).detach()
                else:
                    q = model.encoder_extra_fc2(q[0])
                    k = k[0]
                    q = torch.nn.functional.normalize(q, dim=1)
                    loss = (1 - torch.cosine_similarity(q, k)).mean()
                    sim = torch.cosine_similarity(q, k).detach()

                sample_size = bsz
                return loss, n_tokens, bsz, sample_size, sim

            nonlocal agg_loss, agg_sample_size, agg_logging_output
            if samples is None or len(samples) == 0:
                return
            net_input = samples['net_input']

            tgt_input = copy.deepcopy(net_input)
            if gold_gen:
                tgt_input['src_tokens'] = samples['gold_source']
                tgt_input['lang_tokens'] = net_input['src_lang_tokens']

            else:
                tgt_input['src_tokens'] = net_input['prev_output_tokens']
                tgt_input['lang_tokens'] = net_input['tgt_lang_tokens']

            src_input = copy.deepcopy(net_input)
            src_input['lang_tokens'] = net_input['src_lang_tokens']

            loss, n_tokens, bsz, sample_size, sim = forward_single(tgt_input, src_input)
            if self.args.symmetrical:
                loss2, n_tokens2, bsz2, sample_size2, _ = forward_single(src_input, tgt_input)
                loss = loss + loss2
                n_tokens += n_tokens2
                bsz += bsz2
                sample_size += sample_size2

            # q = model.models[key].encoder(**q_input).encoder_out
            # q = model.encoder_extra_fc(q)
            # q = torch.nn.functional.normalize(q, dim=-1)
            # with torch.no_grad():
            #     net_input['lang_tokens'] = net_input['src_lang_tokens']
            #     k = model.momentum_encoder(**net_input).encoder_out.detach()
            #     k = model.momentum_encoder_extra_fc(k)
            #     k = torch.nn.functional.normalize(k, dim=-1)
            
            # n_tokens, bsz, embed_size = q.shape
            
            # q = model.byol_cross_attention(q, k)
            # loss = (1 - torch.cosine_similarity(q.transpose(0, 1), k.transpose(0, 1), dim=-1)).mean()
            # sample_size = bsz
            logging_output = {
                'loss': utils.item(loss.data),
                'ntokens': n_tokens,
                'nsentences': bsz,
                'sample_size': sample_size,
            }
            if ignore_grad:
                loss *= 0
            else:
                loss *= weight
            optimizer.backward(loss)

            agg_loss += loss.detach().item()
            # TODO make summing of the sample sizes configurable
            agg_sample_size += sample_size
            if logging_output_key not in agg_logging_output:
                agg_logging_output[logging_output_key] = logging_output
            else:
                agg_logging_output[logging_output_key] = {
                    k: agg_logging_output[logging_output_key][k] + logging_output[k] for k in logging_output.keys()
                }
            return sim
        
        def momentum_forward_backward(model, samples, logging_output_key, weight):
            nonlocal agg_loss, agg_sample_size, agg_logging_output
            if samples is None or len(samples) == 0:
                return
            net_input = samples['net_input']
            # key = '{}-{}'.format(net_input['src_lang'], net_input['tgt_lang'])
            key = list(model.models.keys())[0]

            q_input = copy.deepcopy(net_input)
            q_input['src_tokens'] = net_input['prev_output_tokens']
            q_input['lang_tokens'] = net_input['tgt_lang_tokens']
            q = model.models[key].encoder(**q_input).encoder_out[0]
            q = model.encoder_extra_fc(q)
            q = torch.nn.functional.normalize(q, dim=1)

            with torch.no_grad():
                net_input['lang_tokens'] = net_input['src_lang_tokens']
                k = model.momentum_encoder(**net_input).encoder_out[0].detach()
                k = model.momentum_encoder_extra_fc(k)
                k = torch.nn.functional.normalize(k, dim=1)

            bsz, embed_size = q.shape
            l_pos = torch.bmm(q.view(bsz, 1, -1), k.view(bsz, -1, 1)).squeeze(1)
            # model.momentum_bank = model.momentum_bank.to(device=q.device)
            l_neg = torch.mm(q.view(bsz, embed_size), model.momentum_bank.clone().detach())
            logits = torch.cat([l_pos, l_neg], dim=1)
            
            label = torch.zeros(bsz, device=logits.device).long()
            loss = torch.nn.functional.cross_entropy(logits / self.args.momentum_contrast_t, label)
            sample_size = bsz
            logging_output = {
                'loss': utils.item(loss.data),
                'ntokens': sample_size,
                'nsentences': logits.size(0),
                'sample_size': sample_size,
            }
            if ignore_grad:
                loss *= 0
            else:
                loss *= weight
            optimizer.backward(loss)
            # model.momentum_bank.data.copy_(torch.cat([model.momentum_bank[:, bsz:], k.T], dim=1))
            model.dequeue_and_enqueue(k)
            agg_loss += loss.detach().item()
            # TODO make summing of the sample sizes configurable
            agg_sample_size += sample_size
            if logging_output_key not in agg_logging_output:
                agg_logging_output[logging_output_key] = logging_output
            else:
                agg_logging_output[logging_output_key] = {
                    k: agg_logging_output[logging_output_key][k] + logging_output[k] for k in logging_output.keys()
                }

        def pg_forward_backward(sim, action_log_probs, logging_output_key, weight):
            nonlocal agg_loss, agg_sample_size, agg_logging_output
            if action_log_probs is None or len(action_log_probs) == 0 or sim is None:
                return
            bsz, seq_len = action_log_probs.size()
            if sim.dim() == 1:
                loss = - (sim.unsqueeze(1) * action_log_probs).sum(1).mean()
            else:
                loss = - (sim[:, 1: 1+seq_len] * action_log_probs).sum(1).mean()
            
            logging_output = {
                'loss': utils.item(loss.data),
                'ntokens': bsz,
                'nsentences': bsz,
                'sample_size': bsz,
            }
            if ignore_grad:
                loss *= 0
            else:
                loss *= weight

            optimizer.backward(loss)

            agg_loss += loss.detach().item()
            agg_sample_size += bsz
            if logging_output_key not in agg_logging_output:
                agg_logging_output[logging_output_key] = logging_output
            else:
                agg_logging_output[logging_output_key] = {
                    k: agg_logging_output[logging_output_key][k] + logging_output[k] for k in logging_output.keys()
                }
        if self.lambda_otf_bt > 0.0 or self.momentum_contrast_loss_ratio > 0.0 or self.byol_ratio or self.gold_gen_byol_ratio > 0.0 or self.decoder_byol_ratio > 0.0:
            for lang in self.langs:
                sample_key = _get_bt_dataset_key(lang)
                if sample_key not in self.dataset('train').datasets.keys():
                    continue
                real_sample_list, action_log_probs = self.dataset('train').datasets[sample_key].real_collater(sample[sample_key], self.pg_ratio>0.0)
                if real_sample_list is not None:
                    sims = []
                    for real_sample in real_sample_list:
                        if self.lambda_otf_bt > 0.0:
                            forward_backward(model, real_sample, sample_key, self.lambda_otf_bt)
                        if self.momentum_contrast_loss_ratio > 0:
                            momentum_forward_backward(model, real_sample, sample_key.replace('bt', 'momentum'), self.momentum_contrast_loss_ratio)
                        if self.byol_ratio > 0.0:
                            sim = byol_forward_backward(model, real_sample, sample_key.replace('bt', 'byol'), self.byol_ratio)
                            sims.append(sim)
                        if self.gold_gen_byol_ratio > 0.0:
                            byol_forward_backward(model, real_sample, sample_key.replace('bt', 'gold_gen_byol'), self.gold_gen_byol_ratio, gold_gen=True)
                        if self.decoder_byol_ratio > 0.0:
                            decoder_byol_forward_backward(model, real_sample, sample_key.replace('bt', 'decoder_byol'), self.decoder_byol_ratio)
                    if self.pg_ratio > 0:
                        pg_forward_backward(torch.cat(sims, dim=0), action_log_probs, sample_key.replace('bt', 'pg'), self.pg_ratio)

        if self.lambda_denoising > 0.0:
            for lang in self.langs:
                sample_key = _get_denoising_dataset_key(lang)
                if sample_key not in sample:
                    continue
                forward_backward(model, sample[sample_key], sample_key, self.lambda_denoising)

        if self.lambda_bart_pretrain > 0.0:
            for lang in self.langs:
                sample_key = _get_bart_dataset_key(lang)
                if sample_key not in sample:
                    continue
                forward_backward(model, sample[sample_key], sample_key, self.lambda_bart_pretrain)
        
        if self.lambda_parallel > 0.0:
            for sample_key in self.model_lang_pairs:
                forward_backward(model, sample[sample_key], sample_key, self.lambda_parallel)
                if self.parallel_byol_ratio > 0:
                    byol_forward_backward(model, sample[sample_key], sample_key + "parallel_byol", self.parallel_byol_ratio)

        return agg_loss, agg_sample_size, agg_logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            agg_loss, agg_sample_size, agg_logging_output = 0., 0., {}
            for lang_pair in self.eval_lang_pairs:
                if lang_pair not in sample or sample[lang_pair] is None or len(sample[lang_pair]) == 0:
                    continue
                loss, sample_size, logging_output = criterion(model, sample[lang_pair])
                agg_loss += loss.data.item()
                # TODO make summing of the sample sizes configurable
                agg_sample_size += sample_size
                agg_logging_output[lang_pair] = logging_output
        return agg_loss, agg_sample_size, agg_logging_output

    def update_step(self, num_updates):
        def lambda_step_func(config, n_iter):
            """
            Update a lambda value according to its schedule configuration.
            """
            ranges = [i for i in range(len(config) - 1) if config[i][0] <= n_iter < config[i + 1][0]]
            if len(ranges) == 0:
                assert n_iter >= config[-1][0]
                return config[-1][1]
            assert len(ranges) == 1
            i = ranges[0]
            x_a, y_a = config[i]
            x_b, y_b = config[i + 1]
            return y_a + (n_iter - x_a) * float(y_b - y_a) / float(x_b - x_a)

        # if self.lambda_parallel_steps is not None:
        #     self.lambda_parallel = lambda_step_func(self.lambda_parallel_steps, num_updates)
        if self.lambda_denoising_steps is not None:
            self.lambda_denoising = lambda_step_func(self.lambda_denoising_steps, num_updates)
        if self.lambda_otf_bt_steps is not None:
            self.lambda_otf_bt = lambda_step_func(self.lambda_otf_bt_steps, num_updates)
        if self.lambda_bart_pretrain_steps is not None:
            self.lambda_bart_pretrain = lambda_step_func(self.lambda_bart_pretrain_steps, num_updates)
        if self.lambda_parallel_steps is not None:
            self.lambda_parallel = lambda_step_func(self.lambda_parallel, num_updates)

        if self.byol_steps is not None:
            self.byol_ratio = lambda_step_func(self.byol_steps, num_updates)
        if self.decoder_byol_steps is not None:
            self.decoder_byol_ratio = lambda_step_func(self.decoder_byol_steps, num_updates)
        if self.parallel_byol_steps is not None:
            self.parallel_byol_ratio = lambda_step_func(self.parallel_byol_steps, num_updates)
        if self.gold_gen_steps is not None:
            self.gold_gen_byol_ratio = lambda_step_func(self.gold_gen_steps, num_updates)
        if self.moco_steps is not None:
            self.momentum_contrast_loss_ratio = lambda_step_func(self.moco_steps, num_updates)
        if self.pg_steps is not None:
            self.pg_ratio = lambda_step_func(self.pg_steps, num_updates)
        if self.momentum_beta_steps is not None:
            self.momentum_contrast_beta = lambda_step_func(self.momentum_beta_steps, num_updates)
    
    def update_momentum_paras(self, model):
        with torch.no_grad():
            if hasattr(model, 'momentum_encoder'):
                for momentum_para, encoder_para in zip(model.momentum_encoder.parameters(), model.models[model.keys[0]].encoder.parameters()):
                    momentum_para.data.copy_(self.momentum_contrast_beta * momentum_para + (1 - self.momentum_contrast_beta) * encoder_para.detach())
            if hasattr(model, 'momentum_encoder_extra_fc'):
                for momentum_para, encoder_para in zip(model.momentum_encoder_extra_fc.parameters(), model.encoder_extra_fc.parameters()):
                    momentum_para.data.copy_(self.momentum_contrast_beta * momentum_para + (1 - self.momentum_contrast_beta) * encoder_para.detach())
            if hasattr(model, 'momentum_decoder'):
                for momentum_para, decoder_para in zip(model.momentum_decoder.parameters(), model.models[model.keys[0]].decoder.parameters()):
                    momentum_para.data.copy_(self.momentum_contrast_beta * momentum_para + (1 - self.momentum_contrast_beta) * decoder_para.detach())

    def aggregate_logging_outputs(self, logging_outputs, criterion):
        # aggregate logging outputs for each language pair
        logging_output_keys = {
            key
            for logging_output in logging_outputs
            for key in logging_output
        }
        lang_pair_keys = set(self.langs + [
            _get_bt_dataset_key(lang)
            for lang in self.langs
        ] + [
            _get_denoising_dataset_key(lang)
            for lang in self.langs
        ] + [
            _get_bart_dataset_key(lang)
            for lang in self.langs
        ] + [
            _get_bt_dataset_key(lang).replace('bt', 'momentum')
            for lang in self.langs
        ] + [
            _get_bt_dataset_key(lang).replace('bt', 'byol')
            for lang in self.langs
        ] + [
            _get_bt_dataset_key(lang).replace('bt', 'gold_gen_byol')
            for lang in self.langs
        ] + [
            _get_bt_dataset_key(lang).replace('bt', 'pg')
            for lang in self.langs
        ] + [
            sample_key + "parallel_byol" for sample_key in self.model_lang_pairs
        ] + [
            _get_bt_dataset_key(lang).replace('bt', 'decoder_byol')
            for lang in self.langs
        ] + self.eval_lang_pairs + self.model_lang_pairs)
        logging_output_keys = logging_output_keys.intersection(lang_pair_keys)

        for key in logging_output_keys:
            if key.startswith('bt'):
                self.dataset('train').datasets[key].reset()

        return self._aggregate_logging_outputs(logging_outputs, criterion, logging_output_keys)

    def _aggregate_logging_outputs(self, logging_outputs, criterion, logging_output_keys=None):
        logging_output_keys = logging_output_keys
        # aggregate logging outputs for each language pair
        agg_logging_outputs = {
            key: criterion.__class__.aggregate_logging_outputs([
                logging_output.get(key, {}) for logging_output in logging_outputs
            ])
            for key in logging_output_keys
        }

        def sum_over_languages(key):
            return sum(logging_output[key] for logging_output in agg_logging_outputs.values())

        # flatten logging outputs
        flat_logging_output = {
            '{}:{}'.format(lang_pair, k): v
            for lang_pair, agg_logging_output in agg_logging_outputs.items()
            for k, v in agg_logging_output.items()
        }
        flat_logging_output['loss'] = sum_over_languages('loss')
        if any('nll_loss' in logging_output for logging_output in agg_logging_outputs.values()):
            flat_logging_output['nll_loss'] = sum_over_languages('nll_loss')
        flat_logging_output['sample_size'] = sum_over_languages('sample_size')
        flat_logging_output['nsentences'] = sum_over_languages('nsentences')
        flat_logging_output['ntokens'] = sum_over_languages('ntokens')
        return flat_logging_output

    def alter_dataset_langtok(self, lang_pair_dataset,
                              src_eos=None, src_lang=None, tgt_eos=None, tgt_lang=None):
        if not self.args.change_langtok :
            return lang_pair_dataset    
        new_src_eos = None
        new_tgt_bos = self.get_decoder_langtok(tgt_lang)

        # new_src_eos = None
        # if self.args.encoder_langtok is not None and src_eos is not None \
        #    and src_lang is not None and tgt_lang is not None:
        #     new_src_eos = self.get_encoder_langtok(src_lang, tgt_lang)
        # else:
        #     src_eos = None

        # new_tgt_bos = None
        # if self.args.decoder_langtok and tgt_eos is not None and tgt_lang is not None:
        #     new_tgt_bos = self.get_decoder_langtok(tgt_lang)
        # else:
        #     tgt_eos = None

        return TransformEosLangPairDataset(
            lang_pair_dataset,
            src_eos=src_eos,
            new_src_eos=new_src_eos,
            tgt_bos=tgt_eos,
            new_tgt_bos=new_tgt_bos,
        )

    @property
    def source_dictionary(self):
        return self.dict

    @property
    def target_dictionary(self):
        return self.dict

    def get_decoder_langtok(self, tgt_lang):
        return self.dict.index('[{}_bos]'.format(tgt_lang))

    @property
    def generator_para(self):
        return {
            # 'nyt': [1, 0, 512],
            # 'cnndm': [1, 0, 512],
            # 'inshorts': [1, 0, 140],
            self.lang_token_dic['article']: [1, 0, 512],
            self.lang_token_dic['summary']: [1, 0, self.args.gen_sum_len],
        }

    @property
    def source_dictionary(self):
        return self.dict
    
    @property
    def target_dictionary(self):
        return self.dict

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        from fairseq.data import LanguagePairDataset
        return LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary)