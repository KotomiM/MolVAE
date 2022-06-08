###### for vjtnn #####
# for vjtnn preprocess data on MolTree
python tools/preprocess.py --config configs/logp04/vjtnn.json
# for vjtnn training on logp04 training dataset
python tools/train.py --config configs/logp04/vjtnn.json
# for vjtnn decoding test on logp04 validation dataset
python tools/translate.py --config configs/logp04/vjtnn.json --model ckpt/logp04/vjtnn/withoutGAN/vjtnn_LSTM.epoch8
# for vjtnn evaluation on logp04 validation dataset
python tools/eval.py  --config configs/logp04/eval.json --result results/logp04/vjtnn/your_result_file 
