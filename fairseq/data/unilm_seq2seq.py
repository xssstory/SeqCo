
import numpy as np
import torch

from . import data_utils, FairseqDataset


def batch_list_to_batch_tensors(batch):
    batch_tensors = []
    for x in zip(*batch):
        if isinstance(x[0], torch.Tensor):
            batch_tensors.append(torch.stack(x))
        else:
            batch_tensors.append(torch.tensor(x, dtype=torch.long))
    return batch_tensors


class UniLMSeq2SeqDataset(FairseqDataset):
    """A pair of torch.utils.data.Datasets."""

    def __init__(
        self, src, src_sizes,
        tgt=None, tgt_sizes=None,
        tokenizer=None,
        max_source_seq_length=1024, max_target_seq_length=1024,
        shuffle=True,
        random_prob=0.1, keep_prob=0.1,
        mask_way='v2', target_mask_prob=-1.0, num_max_mask_token=0,
        source_mask_prob=-1.0,
        batch_warmup_count=5,
    ):
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer)
        self.max_source_seq_length = max_source_seq_length
        self.max_target_seq_length = max_target_seq_length
        self.shuffle = shuffle

        self.random_prob = random_prob
        self.keep_prob = keep_prob
        self.mask_way = mask_way
        self.target_mask_prob = target_mask_prob
        self.num_max_mask_token = num_max_mask_token
        self.source_mask_prob = source_mask_prob
        
        self.pad_id = tokenizer.pad_token_id
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.mask_id = tokenizer.mask_token_id

        self.batch_warmup_count = batch_warmup_count
        print('batch_warmup_count ', self.batch_warmup_count)
        self.batch_count = 0

    def __getitem__(self, index):
        return {
            'id': index,
            'source': self.src[index],
            'target': self.tgt[index] if self.tgt is not None else None,
        }

    def __len__(self):
        return len(self.src)

    # you must have this for v0.9
    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch."""
        return self.collate(
            samples, self.tokenizer,
            self.max_source_seq_length, self.max_target_seq_length,
        )

    def __trunk(self, ids, max_len, append_sep=True):
        if append_sep:
            max_len -= 1
        if len(ids) > max_len:
            ids = ids[:max_len]
        if append_sep:
            ids = ids + [self.sep_id]
        return ids

    def __pad(self, ids, max_len):
        if len(ids) < max_len:
            return ids + [self.pad_id] * (max_len - len(ids))
        else:
            assert len(ids) == max_len
            return ids

    def get_masked_token(self, tk_id):
        p = np.random.random()
        if p < self.keep_prob:
            return tk_id
        elif p < self.keep_prob + self.random_prob:
            return np.random.randint(0, self.vocab_size - 1)
        else:
            return self.mask_id
    

    def sample2netInput(self, sample_src, sample_tgt, max_source_len, max_target_len):
        source_ids = self.__trunk([self.cls_id] + sample_src, max_source_len, append_sep=self.mask_way != 'v0')
        target_ids = sample_tgt
        if self.mask_way == 'v0':
            target_ids = [self.sep_id] + target_ids
        target_ids = self.__trunk(target_ids, max_target_len, append_sep=self.mask_way != 'v0')

        num_source_tokens = len(source_ids)
        num_target_tokens = len(target_ids)

        if self.source_mask_prob > 0:
            for i in range(num_source_tokens):
                tk_id = source_ids[i]
                if tk_id != self.cls_id and tk_id != self.sep_id:
                    r = np.random.random()
                    if r < self.source_mask_prob:
                        source_ids[i] = self.get_masked_token(tk_id)

        source_ids = self.__pad(source_ids, max_source_len)
        target_ids = self.__pad(target_ids, max_target_len)

        if self.mask_way == 'v0':
            masked_pos = []
            masked_ids = []
            masked_weights = []
            for pos in range(num_target_tokens):
                if pos + 1 != num_target_tokens:
                    masked_ids.append(target_ids[pos + 1])
                else:
                    masked_ids.append(self.sep_id)
                masked_pos.append(pos)
                masked_weights.append(1)

                r = np.random.random()
                if r < self.target_mask_prob and pos > 0:
                    target_ids[pos] = self.get_masked_token(target_ids[pos])
            
            masked_ids = self.__pad(masked_ids, self.num_max_mask_token)
            masked_pos = self.__pad(masked_pos, self.num_max_mask_token)
            masked_weights = self.__pad(masked_weights, self.num_max_mask_token)

            return source_ids, target_ids, masked_ids, masked_pos, masked_weights, num_source_tokens, num_target_tokens
        elif self.mask_way == 'v1':
            masked_pos = list(range(num_target_tokens))
            np.random.shuffle(masked_pos)

            num_masked_token = \
                min(self.num_max_mask_token, int(self.target_mask_prob * num_target_tokens))
            if num_masked_token <= 0:
                num_masked_token = 1

            masked_pos = masked_pos[:num_masked_token]

            masked_ids = []
            masked_weights = []
            for pos in masked_pos:
                masked_ids.append(target_ids[pos])
                target_ids[pos] = self.get_masked_token(target_ids[pos])
                masked_weights.append(1)
            
            masked_ids = self.__pad(masked_ids, self.num_max_mask_token)
            masked_pos = self.__pad(masked_pos, self.num_max_mask_token)
            masked_weights = self.__pad(masked_weights, self.num_max_mask_token)

            return source_ids, target_ids, masked_ids, masked_pos, masked_weights, num_source_tokens, num_target_tokens
        elif self.mask_way == 'v2':
            pseudo_ids = []
            label_ids = []
            for pos in range(num_target_tokens):
                tk_id = target_ids[pos]
                masked_tk_id = self.get_masked_token(tk_id)
                pseudo_ids.append(masked_tk_id)
                label_ids.append(tk_id)
                r = np.random.random()
                if r < self.target_mask_prob:
                    target_ids[pos] = masked_tk_id
            label_ids = self.__pad(label_ids, max_target_len)
            pseudo_ids = self.__pad(pseudo_ids, max_target_len)

            return source_ids, target_ids, label_ids, pseudo_ids, num_source_tokens, num_target_tokens
        elif self.mask_way == 'v3_lang_tag':
            pseudo_ids = []
            label_ids = []
            for pos in range(num_target_tokens):
                tk_id = target_ids[pos]
                masked_tk_id = self.get_masked_token(tk_id)
                pseudo_ids.append(masked_tk_id)
                label_ids.append(tk_id)
                r = np.random.random()
                if pos == 0:
                    target_ids[pos] = tk_id
                else:
                    if r < self.target_mask_prob:
                        target_ids[pos] = masked_tk_id
            label_ids = self.__pad(label_ids, max_target_len)
            pseudo_ids = self.__pad(pseudo_ids, max_target_len)

            return source_ids, target_ids, label_ids, pseudo_ids, num_source_tokens, num_target_tokens


    def collate(self, samples, tokenizer, max_source_seq_length, max_target_seq_length):
        if len(samples) == 0:
            return {}

        id = torch.LongTensor([s['id'] for s in samples])

        self.batch_count += 1
        if self.batch_count <= self.batch_warmup_count:
            max_source_len = max_source_seq_length
            max_target_len = max_target_seq_length
        else:
            max_src_sample = max( [len(sample['source']) for sample in samples] )
            src_delta = 2 if self.mask_way == 'v0' else 1
            max_tgt_sample = max( [len(sample['target']) for sample in samples if sample['target'] is not None] )
            tgt_delta = 1
            max_source_len = min(max_source_seq_length, max_src_sample + src_delta)
            max_target_len = min(max_target_seq_length, max_tgt_sample + tgt_delta)

        batch_list = []
        for sample in samples:
            sample_src = sample['source'].tolist()
            sample_tgt = sample['target'].tolist() if sample['target'] is not None else None
            batch_list.append( self.sample2netInput(sample_src, sample_tgt, max_source_len, max_target_len) )

        source_ids, target_ids, label_ids, pseudo_ids, num_source_tokens, num_target_tokens = batch_list_to_batch_tensors(batch_list)
        ntokens = num_target_tokens.sum().item()
        # ntokens = sum([len(sample['target']) for sample in samples])
        '''
        print('source_ids', source_ids.size())
        print('target_ids', target_ids.size())
        print('label_ids', label_ids.size())
        print('pseudo_ids', pseudo_ids.size())
        print('num_source_tokens', num_source_tokens.size())
        print('num_target_tokens', num_target_tokens.size())
        '''

        '''
        inputs = {'source_ids': batch[0],
                'target_ids': batch[1],
                'label_ids': batch[2], 
                'pseudo_ids': batch[3],
                'num_source_tokens': batch[4],
                'num_target_tokens': batch[5]}
        '''

        return {
            'id': id,
            'ntokens': ntokens,
            'net_input': {
                'source_ids': source_ids,
                'target_ids': target_ids,
                'label_ids': label_ids,
                'pseudo_ids': pseudo_ids,
                'num_source_tokens': num_source_tokens,
                'num_target_tokens': num_target_tokens,
            },
            'target': label_ids,
        }


    # not sure if we still need this
    '''
    def get_dummy_batch(self, num_docs, max_positions, src_len=128, tgt_len=128):
        max_source_positions, max_target_positions = self._get_max_positions(max_positions)
        # src_len, tgt_len = min(src_len, max_source_positions), min(tgt_len, max_target_positions)
        bsz = num_docs

        sent_len = MAX_DOC_LEN // self.max_doc_len
        last_sent_len = MAX_DOC_LEN - (self.max_doc_len-1)*sent_len

        def create_tgt():
            return torch.LongTensor([self.tgt_dict.index('F')] * self.max_doc_len)

        def create_src():
            doc = []
            for i in range(self.max_doc_len):
                cur_sent_len = sent_len if i != self.max_doc_len-1 else last_sent_len
                for j in range(cur_sent_len-1):
                    doc.append(self.src_dict.unk())
                if i != self.max_doc_len-1:
                    doc.append(self.sent_sep_idx)
            return torch.LongTensor(doc)

        batch = self.collater([
            {
                'id': i,
                'source': create_src(),
                'target': create_tgt(),
            }
            for i in range(bsz)
        ])

        return batch
    '''

    def num_tokens(self, index):
        """Return an example's length (number of tokens), used for batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Ordered indices for batching."""
        '''we need random order'''
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        '''
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]
        '''
        return indices

    def valid_size(self, index, max_positions):
        """Check if an example's size is valid according to max_positions."""
        max_source_positions, max_target_positions = self._get_max_positions(max_positions)
        return (
            self.src_sizes[index] <= max_source_positions
            and (self.tgt_sizes is None or self.tgt_sizes[index] <= max_target_positions)
        )

    def _get_max_positions(self, max_positions):
        if max_positions is None:
            return self.max_source_positions, self.max_target_positions
        assert len(max_positions) == 2
        max_src_pos, max_tgt_pos = max_positions
        return min(self.max_source_positions, max_src_pos), min(self.max_target_positions, max_tgt_pos)

    # note that you must have `prefetch` for v0.9
    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False)
            and (getattr(self.tgt, 'supports_prefetch', False) or self.tgt is None)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
