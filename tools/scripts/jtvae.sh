###### for jtvae #####
# for jtvae preprocess data on MolTree
python tools/preprocess.py --config configs/moses/jtvae.json --get_vocab # options
# for jtvae training on moses training dataset
python tools/train.py --config configs/moses/jtvae.json 
# for jtvae decoding test on moses validation dataset
python tools/generate.py --config configs/moses/jtvae.json --model ckpt/moses/jtvae/jtvae_GRU.0
# for jtvae evaluation on moses validation dataset
python tools/eval.py  --config configs/moses/eval.json --result results/moses/jtvae/your_result_file 
