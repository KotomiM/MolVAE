###### for hiervae #####
# for hiervae preprocess data on MolGraph
python tools/preprocess.py --config configs/moses/hiervae.json --get_vocab
# for hiervae training on moses training dataset
python tools/train.py --config configs/moses/hiervae.json 
# for hiervae generating test on moses dataset
python tools/generate.py --config configs/moses/hiervae.json --model ckpt/moses/hiervae/hiervae_LSTM.0
# for hiervae evaluation on moses dataset
python tools/eval.py  --config configs/moses/eval.json --result results/moses/hiervae/your_result_file 
