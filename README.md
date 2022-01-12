

This is an implementation of the SeqCo (**Seq**uence Level **Co**ntrastive Learning for Text Summarizationn) model described in [Sequence Level Contrastive Learning for Text Summarization](https://arxiv.org/abs/2109.03481)

#### Installation

```
pip install pyrouge==0.1.3
pip install pytorch-transformers==1.1.0

# For rouge-1.5.5.pl
sudo apt-get update
sudo apt-get install expat
sudo apt-get install libexpat-dev -y

# For files2fouge
git clone https://github.com/pltrdy/files2rouge.git
cd files2rouge
python setup_rouge.py
python setup.py install
```

#### Prepare the data and pretrained BART model

1. Follow the instruction in https://github.com/abisee/cnn-dailymail to download and process into data-files such that `test.source` and `test.target` has one line for each non-tokenized sample.
2. `sh seqco_scripts/bpe_preprocess.sh`. 
In `seqco_scripts/bpe_preprocess.sh`, you need to change the `--inputs` to the path you store `test.source` and `test.target`
3. `sh binarize.sh`

#### Train the model on CNN/DM dataset

`sh seqco_scripts/train_cnndm.sh`

#### Generate summary

`sh seqco_scripts/infer_cnndm.sh`

#### Compute Rouge

`sh seqco_script/evaluate_sum.sh $PATH_TO_GENERATED_SUMMAY $CHECKPOINT`

#### Reference

If you find our code is useful, please cite the following paper:

```
@article{xu2021sequence,
  title={Sequence Level Contrastive Learning for Text Summarization},
  author={Xu, Shusheng and Zhang, Xingxing and Wu, Yi and Wei, Furu},
  journal={arXiv preprint arXiv:2109.03481},
  year={2021}
}
```





