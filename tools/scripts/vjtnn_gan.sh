###### for vjtnn gan #####
# for vjtnn preprocess data on MolTree
python tools/preprocess.py --config configs/qed/vjtnn_gan.json --get_vocab
# for vjtnn training on logp04 training dataset
python tools/train.py --config configs/qed/vjtnn_gan.json
# for vjtnn decoding test on logp04 validation dataset
python tools/translate.py --config configs/logp04/vjtnn_gan.json --model ckpt/logp04/vjtnn/your_model_file
# for vjtnn evaluation on logp04 validation dataset
python tools/eval.py  --config configs/logp04/eval.json --result results/logp04/vjtnn/your_result_file 
