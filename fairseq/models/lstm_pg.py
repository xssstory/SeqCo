# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from fairseq import options, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import AdaptiveSoftmax
from fairseq.modules import (
    LayerNorm,
)


@register_model('lstm_pg')
class LSTMModel(FairseqEncoderDecoderModel):
    def __init__(self, args, encoder, decoder, task):
        super().__init__(encoder, decoder)

        if (task.momentum_contrast_loss_ratio > 0.0 or task.moco_steps is not None
        or task.byol_ratio > 0.0 or task.byol_steps is not None
        or task.parallel_byol_ratio > 0.0 or task.parallel_byol_steps is not None
        or task.gold_gen_byol_ratio > 0.0 or task.gold_gen_steps is not None
        or task.decoder_byol_ratio > 0.0 or task.decoder_byol_steps is not None):
            self.encoder_extra_fc = nn.Sequential(nn.Linear(args.encoder_hidden_size * 2, args.encoder_hidden_size * 4), nn.ReLU(), nn.Linear(args.encoder_hidden_size * 4, args.encoder_hidden_size * 2))
            if not args.cross_byol:
                self.encoder_extra_fc2 = nn.Sequential(LayerNorm(args.encoder_hidden_size * 2), nn.Linear(args.encoder_hidden_size * 2, args.encoder_hidden_size * 4), nn.ReLU(), nn.Linear(args.encoder_hidden_size * 4, args.encoder_hidden_size * 2))

            self.momentum_encoder = copy.deepcopy(encoder)
            self.momentum_encoder.requires_grad_(False)
            if task.decoder_byol_ratio > 0.0 or task.decoder_byol_steps is not None:
                self.momentum_decoder = copy.deepcopy(decoder)
                self.momentum_decoder.requires_grad_(False)
            self.momentum_encoder_extra_fc = copy.deepcopy(self.encoder_extra_fc)
            self.momentum_encoder_extra_fc.requires_grad_(False)
        if task.momentum_contrast_loss_ratio > 0.0 or task.moco_steps is not None:
            momentum_bank = torch.nn.functional.normalize( torch.randn([args.encoder_embed_dim * 2, args.momentum_contrast_capcity], device='cuda'), dim=0 )
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
                args.encoder_hidden_size * 2,
                8,
                kdim=args.encoder_hidden_size * 2,
                vdim=args.encoder_hidden_size * 2,
                dropout=args.dropout,
                encoder_decoder_attention=True,
            )

        # lang embedding for cnndm and inshorts, 0 for cnndm, 1 for inshorts .

    def byol_cross_attention(self, q, k):
        x, attn = self.cross_attention(
            query=k,
            key=q,
            value=q
        )
        return x

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-freeze-embed', action='store_true',
                            help='freeze encoder embeddings')
        parser.add_argument('--encoder-hidden-size', type=int, metavar='N',
                            help='encoder hidden size')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='number of encoder layers')
        parser.add_argument('--encoder-bidirectional', action='store_true',
                            help='make all layers of encoder bidirectional')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-freeze-embed', action='store_true',
                            help='freeze decoder embeddings')
        parser.add_argument('--decoder-hidden-size', type=int, metavar='N',
                            help='decoder hidden size')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='number of decoder layers')
        parser.add_argument('--decoder-out-embed-dim', type=int, metavar='N',
                            help='decoder output embedding dimension')
        parser.add_argument('--decoder-attention', type=str, metavar='BOOL',
                            help='decoder attention')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion')
        parser.add_argument('--share-decoder-input-output-embed', default=False,
                            action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', default=False, action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')

        # Granular dropout settings (if not specified these default to --dropout)
        parser.add_argument('--encoder-dropout-in', type=float, metavar='D',
                            help='dropout probability for encoder input embedding')
        parser.add_argument('--encoder-dropout-out', type=float, metavar='D',
                            help='dropout probability for encoder output')
        parser.add_argument('--decoder-dropout-in', type=float, metavar='D',
                            help='dropout probability for decoder input embedding')
        parser.add_argument('--decoder-dropout-out', type=float, metavar='D',
                            help='dropout probability for decoder output')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)
        base_architecture(args)

        if args.encoder_layers != args.decoder_layers:
            raise ValueError('--encoder-layers must match --decoder-layers')

        def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
            embed_dict = utils.parse_embedding(embed_path)
            utils.print_embed_overlap(embed_dict, dictionary)
            return utils.load_embedding(embed_dict, dictionary, embed_tokens)

        if args.encoder_embed_path:
            pretrained_encoder_embed = load_pretrained_embedding_from_file(
                args.encoder_embed_path, task.source_dictionary, args.encoder_embed_dim)
        else:
            num_embeddings = len(task.source_dictionary)
            pretrained_encoder_embed = Embedding(
                num_embeddings, args.encoder_embed_dim, task.source_dictionary.pad()
            )

        if args.share_all_embeddings:
            # double check all parameters combinations are valid
            if task.source_dictionary != task.target_dictionary:
                raise ValueError('--share-all-embeddings requires a joint dictionary')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError(
                    '--share-all-embed not compatible with --decoder-embed-path'
                )
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to '
                    'match --decoder-embed-dim'
                )
            pretrained_decoder_embed = pretrained_encoder_embed
            args.share_decoder_input_output_embed = True
        else:
            # separate decoder input embeddings
            pretrained_decoder_embed = None
            if args.decoder_embed_path:
                pretrained_decoder_embed = load_pretrained_embedding_from_file(
                    args.decoder_embed_path,
                    task.target_dictionary,
                    args.decoder_embed_dim
                )
        # one last double check of parameter combinations
        if args.share_decoder_input_output_embed and (
                args.decoder_embed_dim != args.decoder_out_embed_dim):
            raise ValueError(
                '--share-decoder-input-output-embeddings requires '
                '--decoder-embed-dim to match --decoder-out-embed-dim'
            )

        if args.encoder_freeze_embed:
            pretrained_encoder_embed.weight.requires_grad = False
        if args.decoder_freeze_embed:
            pretrained_decoder_embed.weight.requires_grad = False

        encoder = LSTMEncoder(
            dictionary=task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            hidden_size=args.encoder_hidden_size,
            num_layers=args.encoder_layers,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
            bidirectional=args.encoder_bidirectional,
            pretrained_embed=pretrained_encoder_embed,
        )
        decoder = LSTMDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_hidden_size,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            attention=options.eval_bool(args.decoder_attention),
            encoder_output_units=encoder.output_units,
            pretrained_embed=pretrained_decoder_embed,
            share_input_output_embed=args.share_decoder_input_output_embed,
            adaptive_softmax_cutoff=(
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                if args.criterion == 'adaptive_loss' else None
            ),
        )
        return cls(args, encoder, decoder, task)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        """
        Run the forward pass for an encoder-decoder model.

        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., teacher forcing) to
        the decoder to produce the next outputs::

            encoder_out = self.encoder(src_tokens, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)

        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, src_tokens=src_tokens, **kwargs)
        return decoder_out


class LSTMEncoder(FairseqEncoder):
    """LSTM encoder."""
    def __init__(
        self, dictionary, embed_dim=512, hidden_size=512, num_layers=1,
        dropout_in=0.1, dropout_out=0.1, bidirectional=False,
        left_pad=True, pretrained_embed=None, padding_value=0.,
    ):
        super().__init__(dictionary)
        self.num_layers = num_layers
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        num_embeddings = len(dictionary)
        self.padding_idx = dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.lstm = LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.dropout_out if num_layers > 1 else 0.,
            bidirectional=bidirectional,
        )
        self.left_pad = left_pad
        self.padding_value = padding_value

        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2

    def forward(self, src_tokens, src_lengths, **kwargs):
        src_lengths = (src_tokens != self.padding_idx).sum(dim=1)
        if self.left_pad:
            # nn.utils.rnn.pack_padded_sequence requires right-padding;
            # convert left-padding to right-padding
            src_tokens = utils.convert_padding_direction(
                src_tokens,
                self.padding_idx,
                left_to_right=True,
            )

        bsz, seqlen = src_tokens.size()
        seqlen = src_lengths.max()

        # embed tokens
        x = self.embed_tokens(src_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # pack embedded source tokens into a PackedSequence
        packed_x = nn.utils.rnn.pack_padded_sequence(x, src_lengths.data.tolist(), enforce_sorted=False)

        # apply LSTM
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        h0 = x.new_zeros(*state_size)
        c0 = x.new_zeros(*state_size)
        packed_outs, (final_hiddens, final_cells) = self.lstm(packed_x, (h0, c0))

        # unpack outputs and apply dropout
        x, _ = nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=self.padding_value)
        x = F.dropout(x, p=self.dropout_out, training=self.training)
        assert list(x.size()) == [seqlen, bsz, self.output_units]

        if self.bidirectional:

            def combine_bidir(outs):
                out = outs.view(self.num_layers, 2, bsz, -1).transpose(1, 2).contiguous()
                return out.view(self.num_layers, bsz, -1)

            final_hiddens = combine_bidir(final_hiddens)
            final_cells = combine_bidir(final_cells)

        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()

        return {
            'encoder_out': (x, final_hiddens, final_cells),
            'src_tokens': src_tokens,
            'encoder_padding_mask': encoder_padding_mask if encoder_padding_mask.any() else None
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        encoder_out['encoder_out'] = tuple(
            eo.index_select(1, new_order)
            for eo in encoder_out['encoder_out']
        )
        encoder_out['src_tokens'] = encoder_out['src_tokens'].index_select(0, new_order)


        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(1, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return int(1e5)  # an arbitrary large number


class AttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, source_embed_dim, output_embed_dim, bias=False):
        super().__init__()

        self.input_proj = Linear(input_embed_dim, source_embed_dim, bias=bias)
        self.output_proj = Linear(input_embed_dim + source_embed_dim, output_embed_dim, bias=bias)
        self.coverage_proj = Linear(1, source_embed_dim, bias=bias)
        self.v = nn.Linear(source_embed_dim, 1, bias=bias)

    def forward(self, input, source_hids, encoder_padding_mask, coverage):
        # input: bsz x input_embed_dim
        # source_hids: srclen x bsz x source_embed_dim

        # x: bsz x source_embed_dim
        x = self.input_proj(input)
        srclen, bsz, embed_dim = source_hids.size()

        # compute attention
        # attn_scores = (source_hids * x.unsqueeze(0)).sum(dim=2)
        attn_scores = source_hids + x.unsqueeze(0)
        coverage_feature = self.coverage_proj(coverage.view(srclen, bsz, 1))
        attn_scores = attn_scores + coverage_feature

        e = torch.tanh(attn_scores)
        attn_scores = self.v(e).squeeze(dim=-1)

        # don't attend over padding
        if encoder_padding_mask is not None:
            attn_scores = attn_scores.float().masked_fill_(
                encoder_padding_mask,
                float('-inf')
            ).type_as(attn_scores)  # FP16 support: cast to float and back

        attn_scores = F.softmax(attn_scores, dim=0)  # srclen x bsz

        # sum weighted sources
        x = (attn_scores.unsqueeze(2) * source_hids).sum(dim=0)
        coverage = coverage + attn_scores

        x = torch.tanh(self.output_proj(torch.cat((x, input), dim=1)))
        return x, attn_scores, coverage


class LSTMDecoder(FairseqIncrementalDecoder):
    """LSTM decoder."""
    def __init__(
        self, dictionary, embed_dim=512, hidden_size=512, out_embed_dim=512,
        num_layers=1, dropout_in=0.1, dropout_out=0.1, attention=True,
        encoder_output_units=512, pretrained_embed=None,
        share_input_output_embed=False, adaptive_softmax_cutoff=None,
    ):
        super().__init__(dictionary)
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.hidden_size = hidden_size
        self.share_input_output_embed = share_input_output_embed
        self.need_attn = True

        self.adaptive_softmax = None
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.encoder_output_units = encoder_output_units
        if encoder_output_units != hidden_size:
            self.encoder_hidden_proj = Linear(encoder_output_units, hidden_size)
            self.encoder_cell_proj = Linear(encoder_output_units, hidden_size)
        else:
            self.encoder_hidden_proj = self.encoder_cell_proj = None
        self.layers = nn.ModuleList([
            LSTMCell(
                input_size=hidden_size + embed_dim if layer == 0 else hidden_size,
                hidden_size=hidden_size,
            )
            for layer in range(num_layers)
        ])
        if attention:
            # TODO make bias configurable
            self.attention = AttentionLayer(hidden_size, encoder_output_units, hidden_size, bias=False)
        else:
            self.attention = None
        if hidden_size != out_embed_dim:
            self.additional_fc = Linear(hidden_size, out_embed_dim)
        if adaptive_softmax_cutoff is not None:
            # setting adaptive_softmax dropout to dropout_out for now but can be redefined
            self.adaptive_softmax = AdaptiveSoftmax(num_embeddings, hidden_size, adaptive_softmax_cutoff,
                                                    dropout=dropout_out)
        elif not self.share_input_output_embed:
            self.fc_out = Linear(out_embed_dim, num_embeddings, dropout=dropout_out)

        self.p_gen_linear = nn.Linear(hidden_size * 3, 1)

    def forward(self, prev_output_tokens, encoder_out, incremental_state=None, **kwargs):
        src_tokens = encoder_out['src_tokens']
        x, attn_scores, p_gens, coverage = self.extract_features(
            prev_output_tokens, encoder_out, incremental_state
        )

        return self.output_layer(x, p_gens, attn_scores, src_tokens=src_tokens), attn_scores, coverage

    def extract_features(
        self, prev_output_tokens, encoder_out, incremental_state=None
    ):
        """
        Similar to *forward* but only return features.
        """
        encoder_padding_mask = encoder_out['encoder_padding_mask']
        encoder_out = encoder_out['encoder_out']

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
        bsz, seqlen = prev_output_tokens.size()

        # get outputs from encoder
        encoder_outs, encoder_hiddens, encoder_cells = encoder_out[:3]
        srclen = encoder_outs.size(0)

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # initialize previous states (or get from cache during incremental generation)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is not None:
            prev_hiddens, prev_cells, input_feed = cached_state
        else:
            num_layers = len(self.layers)
            prev_hiddens = [encoder_hiddens[i] for i in range(num_layers)]
            prev_cells = [encoder_cells[i] for i in range(num_layers)]
            if self.encoder_hidden_proj is not None:
                prev_hiddens = [self.encoder_hidden_proj(x) for x in prev_hiddens]
                prev_cells = [self.encoder_cell_proj(x) for x in prev_cells]
            input_feed = x.new_zeros(bsz, self.hidden_size)

        attn_scores = x.new_zeros(srclen, seqlen, bsz)
        p_gens = x.new_zeros(bsz, seqlen)
        coverage = x.new_zeros(srclen, seqlen, bsz)
        outs = []
        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            input = torch.cat((x[j, :, :], input_feed), dim=1)

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                input = F.dropout(hidden, p=self.dropout_out, training=self.training)

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            # apply attention using the last layer's hidden state
            if self.attention is not None:
                out, attn_scores[:, j, :], coverage[:, j, :] = self.attention(hidden, encoder_outs, encoder_padding_mask, coverage[:, j - 1, :])
            else:
                out = hidden
            out = F.dropout(out, p=self.dropout_out, training=self.training)

            p_gen_input = torch.cat(
                [out, hidden, input], 1
            )
            p_gen = self.p_gen_linear(p_gen_input) 
            p_gen = torch.sigmoid(p_gen).squeeze(dim=1)
            p_gens[:, j] = p_gen

            # input feeding
            input_feed = out

            # save final output
            outs.append(out)

        # cache previous states (no-op except during incremental generation)
        utils.set_incremental_state(
            self, incremental_state, 'cached_state',
            (prev_hiddens, prev_cells, input_feed),
        )

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hidden_size)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        if hasattr(self, 'additional_fc') and self.adaptive_softmax is None:
            x = self.additional_fc(x)
            x = F.dropout(x, p=self.dropout_out, training=self.training)

        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        # if not self.training and self.need_attn:
        attn_scores = attn_scores.transpose(0, 2)
        coverage = coverage.transpose(0, 2)
        # else:
        #     attn_scores = None
        return x, attn_scores, p_gens, coverage

    def output_layer(self, x, p_gens, attn, src_tokens, **kwargs):
        """Project features to the vocabulary size."""
        # if self.adaptive_softmax is None:
        #     if self.share_input_output_embed:
        #         x = F.linear(x, self.embed_tokens.weight)
        #     else:
        #         x = self.fc_out(x)
        # if self.force_p_gen is not None:
        #     p_gens = self.force_p_gen

        # project back to size of vocabulary
        if self.adaptive_softmax is None:
            if self.share_input_output_embed:
                logits = F.linear(x, self.embed_tokens.weight)
            else:
                logits = self.fc_out(x)
        #     return logits
        # return x

        assert torch.isfinite(logits).all()
        batch_size = logits.shape[0]
        output_length = logits.shape[1]

        assert src_tokens.shape[0] == batch_size
        # if src_tokens.shape[0] != batch_size:
        #     src_tokens = src_tokens.repeat([batch_size // src_tokens.shape[0], 1])
        #     assert src_tokens.shape[0] == batch_size
        src_length = src_tokens.shape[1]

        # The final output distribution will be a mixture of the normal output
        # distribution (softmax of logits) and attention weights.
        gen_dist = F.softmax(logits, dim=-1) * p_gens.unsqueeze(dim=-1)
        attn_dist = attn * (1 - p_gens.unsqueeze(dim=-1))

        index = src_tokens.unsqueeze(dim=1)
        index = index.repeat([1, output_length, 1])
        all_dist =  gen_dist.scatter_add(2, index, attn_dist)
        return torch.log(all_dist)

        # return x

    def reorder_incremental_state(self, incremental_state, new_order):
        super().reorder_incremental_state(incremental_state, new_order)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is None:
            return

        def reorder_state(state):
            if isinstance(state, list):
                return [reorder_state(state_i) for state_i in state]
            return state.index_select(0, new_order)

        new_state = tuple(map(reorder_state, cached_state))
        utils.set_incremental_state(self, incremental_state, 'cached_state', new_state)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return int(1e5)  # an arbitrary large number

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.uniform_(m.weight, -0.1, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def LSTM(input_size, hidden_size, **kwargs):
    m = nn.LSTM(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def LSTMCell(input_size, hidden_size, **kwargs):
    m = nn.LSTMCell(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def Linear(in_features, out_features, bias=True, dropout=0):
    """Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m


@register_model_architecture('lstm_pg', 'lstm_pg')
def base_architecture(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_freeze_embed = getattr(args, 'encoder_freeze_embed', False)
    args.encoder_hidden_size = getattr(args, 'encoder_hidden_size', 256)
    args.encoder_layers = getattr(args, 'encoder_layers', 1)
    args.encoder_bidirectional = getattr(args, 'encoder_bidirectional', True)
    args.encoder_dropout_in = getattr(args, 'encoder_dropout_in', args.dropout)
    args.encoder_dropout_out = getattr(args, 'encoder_dropout_out', args.dropout)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 128)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_freeze_embed = getattr(args, 'decoder_freeze_embed', False)
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', 256)
    args.decoder_layers = getattr(args, 'decoder_layers', 1)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 128)
    args.decoder_attention = getattr(args, 'decoder_attention', '1')
    args.decoder_dropout_in = getattr(args, 'decoder_dropout_in', args.dropout)
    args.decoder_dropout_out = getattr(args, 'decoder_dropout_out', args.dropout)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', '10000,50000,200000')


# @register_model_architecture('ls', 'lstm_pg')
# def lstm_wiseman_iwslt_de_en(args):
#     args.dropout = getattr(args, 'dropout', 0.1)
#     args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
#     args.encoder_dropout_in = getattr(args, 'encoder_dropout_in', 0)
#     args.encoder_dropout_out = getattr(args, 'encoder_dropout_out', 0)
#     args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 128)
#     args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 128)
#     args.decoder_dropout_in = getattr(args, 'decoder_dropout_in', 0)
#     args.decoder_dropout_out = getattr(args, 'decoder_dropout_out', args.dropout)
#     base_architecture(args)
