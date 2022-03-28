# wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
# wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
# wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'

for SPLIT in train val
do
  python -m examples.roberta.multiprocessing_bpe_encoder \
  --encoder-json encoder.json \
  --vocab-bpe vocab.bpe \
  --inputs "data/cnn_dm/$SPLIT.source" \
  --outputs "data/cnn_dm/$SPLIT.bpe.article" \
  --workers 60 \
  --keep-empty;

  python -m examples.roberta.multiprocessing_bpe_encoder \
  --encoder-json encoder.json \
  --vocab-bpe vocab.bpe \
  --inputs "data/cnn_dm/$SPLIT.target" \
  --outputs "data/cnn_dm/$SPLIT.bpe.summary" \
  --workers 60 \
  --keep-empty;
done
