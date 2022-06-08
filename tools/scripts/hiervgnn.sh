###### for hiervgnn #####
# for hiervgnn preprocess data on MolGraph
python tools/preprocess.py --config configs/logp04/hiervgnn.json
# for hiervgnn training on logp04 training dataset
python tools/train.py --config configs/logp04/hiervgnn.json 
# for hiervgnn decoding test on logp04 validation dataset
python tools/translate.py --config configs/logp04/hiervgnn.json --model your_model_path #ckpt/logp04/hiervgnn/hiervgnn_LSTM.0
# for hiervgnn evaluation on logp04 validation dataset
python tools/eval.py  --config configs/logp04/eval.json --result results/logp04/your_result_file 

