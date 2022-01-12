import torch
from fairseq.models.bart import BARTModel, BARTHubInterface
import argparse
import os
from tqdm import tqdm
import random
import shutil
random.seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, default='/home/v-shushengxu/sum_data/bart_new_cnndm/cnn_dm-bin')
parser.add_argument('--raw-valid', type=str, default=None)
parser.add_argument('--raw-test', type=str, default='/home/v-shushengxu/sum_data/bart_new_cnndm/cnn_dm/test')
parser.add_argument('--input-file', type=str, default=None)
parser.add_argument('--output-dir', type=str, default='bart_back_sum_model_dir/back_sum_load_decoder.10000')
parser.add_argument('--model-dir', type=str, default='/home/v-shushengxu/myblob/shusheng/xss_abs_project/back_sum_load_decoder.10000/pt-results/back_sum.2x4x8.lr.3e-05.warm5000.cnndm_mix_extract_not_share_load_decoder/')
# parser.add_argument('--model-dir', type=str, default='/home/v-shushengxu/myblob/shusheng/xss_abs_project/back_sum_load_decoder/pt-results/back_sum.2x4x8.lr.3e-05.warm5000.cnndm_mix_extract_not_share_load_decoder')
parser.add_argument('--checkpoint', type=str, default='5')
parser.add_argument('--source', type=str, default='article')
parser.add_argument('--target', type=str, default='summary')
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--max-gene-length', type=int, default=60)
parser.add_argument('--min-len', type=int, default=10)
parser.add_argument('--lenpen', type=float, default=1)
parser.add_argument('--beam', type=int, default=6)

parser.add_argument('--start-line', type=int, default=1)
parser.add_argument('--stop-line', type=int, default=1e10)
parser.add_argument('--save-source', action='store_true')

args = parser.parse_args()
print(args)
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
os.system('cp -r {} {}'.format(args.model_dir+"/*.sh", args.output_dir))
os.system('echo {} > {}'.format(args.model_dir, os.path.join(args.output_dir, 'model_path')))

from fairseq import hub_utils
x = hub_utils.from_pretrained(
    model_name_or_path=args.model_dir,
    checkpoint_file='checkpoint{}.pt'.format(args.checkpoint),
    data_name_or_path=args.data_path,
    bpe='gpt2',
    load_checkpoint_heads=True,
    source_lang=args.source,
    target_lang=args.target,
)
bart = BARTHubInterface(x['args'], x['task'], x['models'][0].models['{}-{}'.format(args.source, args.target)])
# bart = BARTModel.from_pretrained(
#     args.model_dir,
#     checkpoint_file='checkpoint{}.pt'.format(args.checkpoint),
#     data_name_or_path=args.data_path,
#     source_lang=args.source,
#     target_lang=args.target,
# )

bart.cuda()
bart.eval()
bart.half()
count = 1
bsz = args.batch_size

for split in ['test', 'validation']:
    if split == 'test':
        source_file = '.'.join([args.raw_test, args.source]) if args.input_file is None else args.input_file
        print(source_file)
    elif args.raw_valid is None:
        continue
    else:
        source_file = '.'.join([args.raw_valid, args.source])

    target_file = os.path.join(args.output_dir, '{}.{}.hypo'.format(split, args.checkpoint))
    if os.path.exists(target_file):
        print("{} already exists !".format(target_file))
        continue
    if args.save_source:
        save_source_file = os.path.join(args.output_dir, '{}.source'.split)
        save_source_file = open(save_source_file, 'w')

    with open(source_file) as source, open(target_file, 'w') as fout:
        # sline = source.readline().strip()
        # slines = [sline]
        slines = []
        for sline in tqdm(source):
            if count > args.stop_line:
                break
            if slines and len(slines) % bsz == 0 and count > args.start_line:
                with torch.no_grad():
                    hypotheses_batch = bart.sample(slines, beam=args.beam, lenpen=args.lenpen, max_len_b=args.max_gene_length, min_len=args.min_len, no_repeat_ngram_size=3)

                for hypothesis in hypotheses_batch:
                    fout.write(hypothesis + '\n')
                    fout.flush()
                slines = []
            if count >= args.start_line:
                slines.append(sline.strip())
                if args.save_source:
                    save_source_file.write(sline.strip() + '\n')
                    save_source_file.flush()
            count += 1

        if slines != []:
            hypotheses_batch = bart.sample(slines, beam=args.beam, lenpen=args.lenpen, max_len_b=args.max_gene_length, min_len=args.min_len, no_repeat_ngram_size=3)
            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + '\n')
                fout.flush()

    if hasattr(locals, 'save_source_file'):
        save_source_file.close()
print(args.output_dir)
