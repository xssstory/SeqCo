
src=article
tgt=summary
export NCCL_LL_THRESHOLD=0
nvidia-smi
export PT_DATA_DIR=/data/msra/sum_data/

datadir=data/cnn_dm-bin

byol_ratio=0.5
decoder_byol=0
gold_gen_byol_ratio=0
parallel_byol_ratio=0
beta=0.99
moco_ratio=0
LR=4e-05
WARMUP_UPDATES=1000
UPDATE_FREQ=8
symmetrical=True
cross_byol=True
TOTAL_NUM_UPDATES=20000

modeldir=resutls/cnndm_models/
BART_FILE=$PT_DATA_DIR/cnndm_bart/bart.large/model.pt

mkdir -vp $modeldir
cp $0 $modeldir
pip install pytorch-transformers==1.1.0 --user
pip install scikit-learn --user
pip install transformers==2.3.0 --user
pip install tensorboardX==1.8 --user
pip install scikit-learn==0.20.3 --user
pip install boto3==1.14.11 --user
echo "**** install boto3==1.14.11 done!"
pip install sentencepiece==0.1.91 --user
echo "**** sentencepiece==0.1.91 done!"

python train.py $datadir \
    --max-sentences=2 \
    --task finetune_summarization \
    --save-dir=$modeldir \
    --source-lang $src --target-lang $tgt \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --share-encoders \
    --load-decoders \
    --lambda-parallel-config=1 \
    --lambda-denoising-config=0 \
    --lambda-otf-bt-config=0 \
    --momentum-contrast-loss-ratio=$moco_ratio \
    --byol-ratio=$byol_ratio \
    --decoder-byol=$decoder_byol \
    --momentum-contrast-beta=$beta \
    --parallel-byol-ratio=$parallel_byol_ratio \
    --gold-gen-byol=$gold_gen_byol_ratio \
    --symmetrical=$symmetrical \
    --cross-byol=$cross_byol \
    --pg-ratio=0 \
    --init-from-pretrained-doc-model \
    --pretrained-doc-model-path=$BART_FILE \
    --required-batch-size-multiple 1 \
    --arch backsum_transformer_bart_large \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --num-workers=0 \
    --find-unused-parameters \
    --ddp-backend=no_c10d \
    --log-interval=50 2>&1 | tee $modeldir/log.txt
