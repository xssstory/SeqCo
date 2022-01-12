# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os

from collections import OrderedDict

from fairseq import utils, distributed_utils
from fairseq.models import (
    FairseqEncoderDecoderModel,
    BaseFairseqModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    base_architecture,
    Embedding,
    TransformerModel,
    FairseqEncoder,
    FairseqIncrementalDecoder,
    EncoderOut,
)

from fairseq.modules import (
    AdaptiveSoftmax,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    try:
        tensors_gather = [torch.zeros_like(tensor)
            for _ in range(torch.distributed.get_world_size())]
        # rank = torch.distributed.get_rank()
        # tensors_gather[rank] = tensor
        # output = torch.cat(tensors_gather, dim=0)
        # torch.distributed.all_reduce(output)
        # print(rank, output[0], force=True)
        torch.distributed.all_gather(tensors_gather, tensor.detach(), async_op=False)
        output = torch.cat(tensors_gather, dim=0)
        return output
    except AssertionError as e:
        print('|WARNING :', e)
        return tensor



@register_model('backsum_transformer')
class BacksumTransformerModel(BaseFairseqModel):
    """Train Transformer models for multiple language pairs simultaneously.

    Requires `--task multilingual_translation`.

    We inherit all arguments from TransformerModel and assume that all language
    pairs use a single Transformer architecture. In addition, we provide several
    options that are specific to the multilingual setting.

    Args:
        --share-encoder-embeddings: share encoder embeddings across all source languages
        --share-decoder-embeddings: share decoder embeddings across all target languages
        --share-encoders: share all encoder params (incl. embeddings) across all source languages
        --share-decoders: share all decoder params (incl. embeddings) across all target languages
    """

    def __init__(self, args, encoders, decoders, task):
        super().__init__()
        assert encoders.keys() == decoders.keys()
        self.keys = list(encoders.keys())
        for key in self.keys:
            assert isinstance(encoders[key], FairseqEncoder)
            assert isinstance(decoders[key], FairseqIncrementalDecoder)

        self.models = nn.ModuleDict({
            key: FairseqEncoderDecoderModel(encoders[key], decoders[key])
            for key in self.keys
        })
        assert args.share_encoders
        if (task.momentum_contrast_loss_ratio > 0.0 or task.moco_steps is not None
        or task.byol_ratio > 0.0 or task.byol_steps is not None
        or task.parallel_byol_ratio > 0.0 or task.parallel_byol_steps is not None
        or task.gold_gen_byol_ratio > 0.0 or task.gold_gen_steps is not None
        or task.decoder_byol_ratio > 0.0 or task.decoder_byol_steps is not None):
            self.encoder_extra_fc = nn.Sequential(nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim * 4), nn.ReLU(), nn.Linear(args.encoder_embed_dim * 4, args.encoder_embed_dim))
            if not args.cross_byol:
                self.encoder_extra_fc2 = nn.Sequential(LayerNorm(args.encoder_embed_dim), nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim * 4), nn.ReLU(), nn.Linear(args.encoder_embed_dim * 4, args.encoder_embed_dim))

            self.momentum_encoder = copy.deepcopy(encoders[self.keys[0]])
            self.momentum_encoder.requires_grad_(False)
            if task.decoder_byol_ratio > 0.0 or task.decoder_byol_steps is not None:
                self.momentum_decoder = copy.deepcopy(decoders[self.keys[0]])
                self.momentum_decoder.requires_grad_(False)
            self.momentum_encoder_extra_fc = copy.deepcopy(self.encoder_extra_fc)
            self.momentum_encoder_extra_fc.requires_grad_(False)
        if task.momentum_contrast_loss_ratio > 0.0 or task.moco_steps is not None:
            momentum_bank = torch.nn.functional.normalize( torch.randn([args.encoder_embed_dim, args.momentum_contrast_capcity], device='cuda'), dim=0 )
            if args.fp16:
                momentum_bank = momentum_bank.half()

            self.register_buffer('momentum_bank', momentum_bank)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        if (task.byol_ratio > 0.0 or task.byol_steps is not None
        or task.parallel_byol_ratio or task.parallel_byol_steps is not None
        or task.gold_gen_byol_ratio or task.gold_gen_steps is not None
        or task.decoder_byol_ratio > 0.0 or task.decoder_byol_steps is not None):
            from fairseq.modules.multihead_attention import MultiheadAttention
            self.cross_attention = MultiheadAttention(
                args.encoder_embed_dim,
                args.encoder_attention_heads,
                kdim=args.encoder_embed_dim,
                vdim=args.encoder_embed_dim,
                dropout=args.attention_dropout,
                encoder_decoder_attention=True,
            )

        # lang embedding for cnndm and inshorts, 0 for cnndm, 1 for inshorts .
        if args.share_decoders and args.share_encoders:
            self.lang_embed_token = Embedding(len(args.langs), args.encoder_embed_dim, padding_idx=None)
            for k in self.models.keys():
                self.models[k].encoder.lang_embed_token = self.lang_embed_token
                self.models[k].decoder.lang_embed_token = self.lang_embed_token
    
    @torch.no_grad()
    def dequeue_and_enqueue(self, keys):
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        momentum_contrast_capcity = self.momentum_bank.shape[1]
        # if ptr + batch_size > momentum_contrast_capcity:
        #     tail = momentum_contrast_capcity - ptr
        #     head = ptr + batch_size - momentum_contrast_capcity
        #     self.momentum_bank[:, ptr: ] = keys[:tail].T
        #     self.momentum_bank[:, :head] = keys[-head:].T 
        # else:
        # # replace the keys at ptr (dequeue and enqueue)
        #     self.momentum_bank[:, ptr:ptr + batch_size] = keys.T
        self.momentum_bank.copy_(torch.cat([self.momentum_bank[:, batch_size:], keys.T], dim=1))
        ptr = (ptr + batch_size) % momentum_contrast_capcity # move pointer

        self.queue_ptr[0] = ptr
    
    def byol_cross_attention(self, q, k):
        x, attn = self.cross_attention(
            query=k,
            key=q,
            value=q
        )
        return x

    def forward(self, src_tokens, src_lengths, prev_output_tokens, src_lang, tgt_lang, src_lang_tokens=None, tgt_lang_tokens=None, **kwargs):
        key = '{}-{}'.format(src_lang, tgt_lang)
        encoder_out = self.models[key].encoder(src_tokens, src_lengths, lang_tokens=src_lang_tokens, **kwargs)
        decoder_out = self.models[key].decoder(
                prev_output_tokens, encoder_out, lang_tokens=tgt_lang_tokens, **kwargs,
            )
        decoder_out[1]['model_key'] = key
        return decoder_out

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        key = net_output[1]['model_key']
        return self.models[key].get_normalized_probs(net_output, log_probs, sample)
    


    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        TransformerModel.add_args(parser)
        parser.add_argument('--share-encoder-embeddings', action='store_true',
                            help='share encoder embeddings across languages')
        parser.add_argument('--share-decoder-embeddings', action='store_true',
                            help='share decoder embeddings across languages')
        parser.add_argument('--share-encoders', action='store_true',
                            help='share encoders across languages')
        parser.add_argument('--share-decoders', action='store_true',
                            help='share decoders across languages')
        parser.add_argument('--load-decoders', action='store_true')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_multilingual_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        # src_langs = [lang_pair.split('-')[0] for lang_pair in task.model_lang_pairs]
        # tgt_langs = [lang_pair.split('-')[1] for lang_pair in task.model_lang_pairs]
        src_langs = task.langs
        tgt_langs = task.langs

        if args.share_encoders:
            args.share_encoder_embeddings = True
        if args.share_decoders:
            args.share_decoder_embeddings = True

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        # build shared embeddings (if applicable)
        shared_encoder_embed_tokens, shared_decoder_embed_tokens = None, None
        if args.share_all_embeddings:
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            shared_encoder_embed_tokens = build_embedding(
                task.dict, 
                embed_dim=args.encoder_embed_dim,
                path=args.encoder_embed_path,
            )
            shared_decoder_embed_tokens = shared_encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            if args.share_encoder_embeddings:
                shared_encoder_embed_tokens = (
                    build_embedding(
                        task.dict,
                        embed_dim=args.encoder_embed_dim,
                        path=args.encoder_embed_path,
                    )
                )
            if args.share_decoder_embeddings:
                shared_decoder_embed_tokens = (
                    build_embedding(
                        task.dict,
                        embed_dim=args.decoder_embed_dim,
                        path=args.decoder_embed_path,
                    )
                )

        # encoders/decoders for each language
        lang_encoders, lang_decoders = {}, {}

        def get_encoder(lang):
            if lang not in lang_encoders:
                if shared_encoder_embed_tokens is not None:
                    encoder_embed_tokens = shared_encoder_embed_tokens
                else:
                    encoder_embed_tokens = build_embedding(
                        task.dict, args.encoder_embed_dim, args.encoder_embed_path
                    )
                lang_encoders[lang] = TransformerEncoder(args, task.dict, encoder_embed_tokens)
            return lang_encoders[lang]

        def get_decoder(lang):
            if lang not in lang_decoders:
                if shared_decoder_embed_tokens is not None:
                    decoder_embed_tokens = shared_decoder_embed_tokens
                else:
                    decoder_embed_tokens = build_embedding(
                        task.dict, args.decoder_embed_dim, args.decoder_embed_path
                    )
                lang_decoders[lang] = TransformerDecoder(args, task.dict, decoder_embed_tokens)
            return lang_decoders[lang]

        # shared encoders/decoders (if applicable)
        shared_encoder, shared_decoder = None, None
        if args.share_encoders:
            shared_encoder = get_encoder(src_langs[0])
        if args.share_decoders:
            shared_decoder = get_decoder(tgt_langs[0])

        encoders, decoders = OrderedDict(), OrderedDict()
        for lang_pair in task.model_lang_pairs:
            src, tgt = lang_pair.split('-')
            encoders[lang_pair] = shared_encoder if shared_encoder is not None else get_encoder(src)
            decoders[lang_pair] = shared_decoder if shared_decoder is not None else get_decoder(tgt)

        return cls(args, encoders, decoders, task)

    def load_state_dict(self, state_dict, strict=True, args=None):
        models_key = [k for k in state_dict.keys() if k.startswith('models')]
        if len(models_key) != 0:
            state_dict_models = state_dict.copy()
            for k, _ in state_dict.items():
                if not k.startswith('models.'):
                    assert k.startswith('lang_embed_token.') or k.startswith('momentum') or k.startswith('encoder_extra_fc') or k == "queue_ptr" or k.startswith("cross_attention")
                    continue
                assert k.startswith('models.')
                lang_pair = k.split('.')[1]
                if lang_pair not in self.models:
                    del state_dict_models[k]
            # for k, _ in state_dict.items():
            #     for lang_pair in self.models:
            #         if k.startswith('models.{}.encoder'.format(lang_pair)):
            #             new_k = k.replace('models.{}.encoder'.format(lang_pair), 'momentum_encoders.{}'.format(lang_pair))
            #             if new_k not in state_dict:
            #                 state_dict_models[new_k] = state_dict[k]
            if "encoder_extra_fc2.3.weight" not in state_dict_models:
                for k in ["encoder_extra_fc2.{}.weight".format(i) for i in range(3)]:
                    if k in state_dict_models:
                        del state_dict_models[k]
                        del state_dict_models[k.replace('weight', "bias")]
            missing_keys, unexpected_keys = super().load_state_dict(state_dict_models, strict=False)
            # assert unexpected_keys == []
            print("----------------------- unexpected_keys --------------")
            print(unexpected_keys)
            print("----------------------missing_keys-------------------")
            print(missing_keys)
            for key in unexpected_keys:
                assert key.startswith('encoder_extra_fc2')
            for key in missing_keys:
                assert key.startswith('momentum') or key.startswith('encoder_extra_fc') or key == "queue_ptr" or key.startswith("cross_attention")
            if missing_keys != []:
                self.momentum_encoder.load_state_dict(self.models[self.keys[0]].encoder.state_dict())
        else: 
            state_dict_subset = state_dict.copy()
            print('Load {} parameters'.format(len(state_dict_subset)))
            if not args.share_encoders:
                encoder_key = [k for k in state_dict_subset.keys() if k.startswith('encoder')]
                for key in encoder_key:
                    del state_dict_subset[key]
            if not args.share_decoders and not args.load_decoders:
                decoder_key = [k for k in state_dict_subset.keys() if k.startswith('decoder')]
                for key in decoder_key:
                    del state_dict_subset[key]
            print('Reserve {} parameters'.format(len(state_dict_subset)))
            if not "encoder.lang_embed_token.weight" in state_dict_subset and hasattr(self, 'lang_embed_token'):
                state_dict_subset["encoder.lang_embed_token.weight"] = self.lang_embed_token.weight
            if not "decoder.lang_embed_token.weight" in state_dict_subset and hasattr(self, 'lang_embed_token'):
                state_dict_subset["decoder.lang_embed_token.weight"] = self.lang_embed_token.weight
            for k, m in self.models.items():
                missing_keys, unexpected_keys = m.load_state_dict(state_dict_subset, False)
                if args.share_encoders and args.share_decoders:
                    assert missing_keys + unexpected_keys == []
                    break
            
            if hasattr(self, "momentum_encoder"):
                encoder_state = self.models[self.keys[0]].encoder.state_dict()
                encoder_state.pop('lang_embed_token.weight', None)
                self.momentum_encoder.load_state_dict(encoder_state)


class TransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([3]))

        self.dropout = args.dropout
        self.encoder_layerdrop = args.encoder_layerdrop

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, embed_dim, self.padding_idx,
            learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layer_wise_attention = getattr(args, 'layer_wise_attention', False)

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(args)
            for i in range(args.encoder_layers)
        ])

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None
        if getattr(args, 'layernorm_embedding', False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

    def forward_embedding(self, src_tokens, lang_tokens):
        # embed tokens and positions
        x = embed = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        # add lang_embedding
        if lang_tokens is not None and hasattr(self, 'lang_embed_token'):
            x.add_(self.lang_embed_token(src_tokens.clone().fill_(lang_tokens)))
        if self.layernorm_embedding:
            x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x, embed

    def forward(self, src_tokens, src_lengths, cls_input=None, lang_tokens=None, return_all_hiddens=False, **unused):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        if self.layer_wise_attention:
            return_all_hiddens = True

        x, encoder_embedding = self.forward_embedding(src_tokens, lang_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        for layer in self.layers:
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if not self.training or (dropout_probability > self.encoder_layerdrop):
                x = layer(x, encoder_padding_mask)
                if return_all_hiddens:
                    encoder_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)
            if return_all_hiddens:
                encoder_states[-1] = x

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
        )

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out.encoder_out is not None:
            encoder_out = encoder_out._replace(
                encoder_out=encoder_out.encoder_out.index_select(1, new_order)
            )
        if encoder_out.encoder_padding_mask is not None:
            encoder_out = encoder_out._replace(
                encoder_padding_mask=encoder_out.encoder_padding_mask.index_select(0, new_order)
            )
        if encoder_out.encoder_embedding is not None:
            encoder_out = encoder_out._replace(
                encoder_embedding=encoder_out.encoder_embedding.index_select(0, new_order)
            )
        if encoder_out.encoder_states is not None:
            for idx, state in enumerate(encoder_out.encoder_states):
                encoder_out.encoder_states[idx] = state.index_select(1, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
            if self._future_mask.size(0) < dim:
                self._future_mask = torch.triu(utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                print('deleting {0}'.format(weights_key))
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)
        for i in range(len(self.layers)):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(state_dict, "{}.layers.{}".format(name, i))

        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


class TransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([3]))

        self.dropout = args.dropout
        self.decoder_layerdrop = args.decoder_layerdrop
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.project_in_dim = Linear(input_embed_dim, embed_dim, bias=False) if embed_dim != input_embed_dim else None

        self.embed_positions = PositionalEmbedding(
            args.max_target_positions, embed_dim, self.padding_idx,
            learned=args.decoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.cross_self_attention = getattr(args, 'cross_self_attention', False)
        self.layer_wise_attention = getattr(args, 'layer_wise_attention', False)

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(args, no_encoder_attn)
            for _ in range(args.decoder_layers)
        ])

        self.adaptive_softmax = None

        self.project_out_dim = Linear(embed_dim, self.output_embed_dim, bias=False) \
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights else None

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), self.output_embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim ** -0.5)

        if args.decoder_normalize_before and not getattr(args, 'no_decoder_final_norm', False):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None
        if getattr(args, 'layernorm_embedding', False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

    def forward(
        self,
        prev_output_tokens,
        encoder_out=None,
        incremental_state=None,
        features_only=False,
        lang_tokens=None,
        **extra_args
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            lang_tokens=lang_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            **extra_args
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out=None,
        incremental_state=None,
        full_context_alignment=False,
        alignment_layer=None,
        alignment_heads=None,
        lang_tokens=None,
        **unused,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer = len(self.layers) - 1

        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:].clone()
            if positions is not None:
                positions = positions[:, -1:].clone()

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        # add lang embedding
        if lang_tokens is not None and hasattr(self, 'lang_embed_token'):
            x += self.lang_embed_token(prev_output_tokens.clone().fill_(lang_tokens))

        if self.layernorm_embedding:
            x = self.layernorm_embedding(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn = None
        inner_states = [x]
        for idx, layer in enumerate(self.layers):
            encoder_state = None
            if encoder_out is not None:
                if self.layer_wise_attention:
                    encoder_state = encoder_out.encoder_states[idx]
                else:
                    encoder_state = encoder_out.encoder_out

            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if not self.training or (dropout_probability > self.decoder_layerdrop):
                x, layer_attn = layer(
                    x,
                    encoder_state,
                    encoder_out.encoder_padding_mask if encoder_out is not None else None,
                    incremental_state,
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=self_attn_padding_mask,
                    need_attn=(idx == alignment_layer),
                    need_head_weights=(idx == alignment_layer),
                )
                inner_states.append(x)
                if layer_attn is not None and idx == alignment_layer:
                    attn = layer_attn.float()

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {'attn': attn, 'inner_states': inner_states}

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                return F.linear(features, self.embed_tokens.weight)
            else:
                return F.linear(features, self.embed_out)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions())

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
            not hasattr(self, '_future_mask')
            or self._future_mask is None
            or self._future_mask.device != tensor.device
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)

        for i in range(len(self.layers)):
            # update layer norms
            layer_norm_map = {
                '0': 'self_attn_layer_norm',
                '1': 'encoder_attn_layer_norm',
                '2': 'final_layer_norm'
            }
            for old, new in layer_norm_map.items():
                for m in ('weight', 'bias'):
                    k = '{}.layers.{}.layer_norms.{}.{}'.format(name, i, old, m)
                    if k in state_dict:
                        state_dict['{}.layers.{}.{}.{}'.format(name, i, new, m)] = state_dict[k]
                        del state_dict[k]

        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict




@register_model_architecture('backsum_transformer', 'backsum_transformer')
def base_multilingual_architecture(args):
    base_architecture(args)
    args.share_encoder_embeddings = getattr(args, 'share_encoder_embeddings', False)
    args.share_decoder_embeddings = getattr(args, 'share_decoder_embeddings', False)
    args.share_encoders = getattr(args, 'share_encoders', False)
    args.share_decoders = getattr(args, 'share_decoders', False)


@register_model_architecture('backsum_transformer', 'backsum_transformer_bart_large')
def multilingual_transformer_bart_large(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4*1024)
    args.encoder_layers = getattr(args, 'encoder_layers', 12)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', True)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 12)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', True)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.max_target_positions = getattr(args, 'max_target_positions', 1024)
    args.max_source_positions = getattr(args, 'max_source_positions', 1024)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', True)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, 'no_scale_embedding', True)
    args.layernorm_embedding = getattr(args, 'layernorm_embedding', True)

    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    args.pooler_activation_fn = getattr(args, 'pooler_activation_fn', 'tanh')
    args.pooler_dropout = getattr(args, 'pooler_dropout', 0.0)
    base_multilingual_architecture(args)
