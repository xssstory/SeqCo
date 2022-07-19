
from .dictionary import Dictionary
import os, torch
from collections import OrderedDict

class TokenizerDictionary(Dictionary):

    def __init__(self, tokenizer):
        self.symbols = []
        self.count = []
        self.indices = OrderedDict()

        self.tokenizer = tokenizer

        dict_size = len(tokenizer)
        for idx in range(dict_size):
            token = self.tokenizer.convert_ids_to_tokens(idx)
            count = 1 # no count information available
            self.indices[token] = len(self.symbols)
            self.symbols.append(token)
            self.count.append(count)
        
        print( '[TokenizerDictionary] size {}'.format( len(tokenizer) ) )
        self.find_special_tokens(tokenizer)


    def find_special_tokens(self, tokenizer):
        self.pad_word = tokenizer.pad_token
        self.pad_index = self.indices[self.pad_word]
        assert tokenizer.pad_token_id == self.pad_index

        self.unk_word = tokenizer.unk_token
        self.unk_index = self.indices[self.unk_word]
        assert tokenizer.unk_token_id == self.unk_index

        self.cls_word = tokenizer.cls_token
        self.cls_index = self.indices[self.cls_word]
        assert tokenizer.cls_token_id == self.cls_index

        self.sep_word = tokenizer.sep_token
        self.sep_index = self.indices[self.sep_word]
        assert tokenizer.sep_token_id == self.sep_index

        self.mask_word = tokenizer.mask_token
        self.mask_index = self.indices[self.mask_word]
        assert tokenizer.mask_token_id == self.mask_index

        # begin of sentence and end of sentence
        self.bos_word = tokenizer.cls_token
        self.bos_index = self.indices[self.bos_word]
        self.eos_word = tokenizer.sep_token
        self.eos_index = self.indices[self.eos_word]

