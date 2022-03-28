export CLASSPATH=stanford-corenlp-full-2016-10-31/stanford-corenlp-3.7.0.jar

base_dir=$1
checkpoint=$2

cat data/cnn_dm/test.target | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > $base_dir/test.target.tokenized
cat $base_dir/test.$checkpoint.hypo | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > $base_dir/test.$checkpoint.tokenized
files2rouge $base_dir/test.$checkpoint.tokenized $base_dir/test.target.tokenized -s $base_dir/test.$checkpoint.rouge --ignore_empty
