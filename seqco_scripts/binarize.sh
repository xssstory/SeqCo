python preprocess.py \
  --source-lang "article" \
  --target-lang "summary" \
  --trainpref "data/cnn_dm/train.bpe" \
  --validpref "data/cnn_dm/val.bpe" \
  --destdir "data/cnn_dm-bin/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;
