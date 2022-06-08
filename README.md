# VAE-based molecule generation and translation

This is a joint PyTorch implementation of three papers in VAE-based molecule generation and translation. The papers and the official repos are as follows:

* [Junction Tree Variational Autoencoder for Molecular Graph Generation (ICML 2018)](https://github.com/wengong-jin/icml18-jtnn)
* [Learning Multimodal Graph-to-Graph Translation for Molecular Optimization (ICLR 2019)](https://github.com/wengong-jin/iclr19-graph2graph)
* [Hierarchical Generation of Molecular Graphs using Structural Motifs (ICML 2020)](https://github.com/wengong-jin/hgraph2graph)

The master branch works with PyTorch 1.8+.

MolVAE has been tested under Python 3.7 with PyTorch 1.11 on cuda 11.4

## Installation

1. Create an Anaconda environment

   ```bash
   conda create --name vae_py37 python=3.7
   conda activate vae_py37
   ```

2. Install RDKit

   ```bash
   conda install rdkit -c rdkit
   ```
   
3. Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g. PyTorch on GPU platforms:    

    ```bash
    conda install pytorch torchvision -c pytorch
    ```

4. Install other requirements:

   ```bash
   pip install -r requirements.txt
   ```

5. Install Chemprop (from source, additional dependency for property-guided finetuning)

   ```bash
   git clone https://github.com/chemprop/chemprop.git
   cd chemprop
   pip install -e .
   ```

## Data Format

* For molecule generation, each line of a training file is a molecule in SMILES representation. 
  * `benchmark/moses` and `benchmark/polymers` are used for generation.
* For molecule translation, each line of a training file is a pair of molecules (molA, molB). The target is to translate from molA towards molB, as molB has better chemical properties. 
  * `benchmark/drd2`, `benchmark/logp04`, `benchmark/logp06` and `benchmark/qed` are used for translation.

## Training

1. Select config file and raw data according to task and appraoch.

   * For molecule generation, go to `configs/moses` or `configs/polymers`.
     * For junction tree approach, use `configs/*/jtvae.json`.
     * For hierarchical substructure approach, use `configs/*/hiervae.json`.
   * For molecule translation, go to `configs/drd2` , `configs/logp04`, `configs/logp06` or `configs/qed`
     * For junction tree approach, according to with or without GAN loss, use `config/*/vjtnn_gan.json` or `configs/*/vjtnn.json`
     * For hierarchical substructure approach, use `configs/*/hiervgnn.json`

2. Extract vocabularies from a given set of molecules and preprocess training data. Add the `--get_vocab` argument if you have not extracted the vocabulary before. Replace `xxx` with your selected json file.

   ```bash
   python tools/preprocess.py --config configs/xxx
   ```

3. Train the model

   * Without GAN loss

     ```bash
     python tools/train.py --config configs/xxx
     ```

   * With GAN loss (only for junction tree approach for molecule translation)

     ```bash
     python tools/train_gan.py --config configs/xxx
     ```

## Testing

* For molecule generation, replace `yyy` with your selected model in `ckpt/moses` or `ckpt/polymers`.

  ```bash
  python tools/generate.py --config configs/xxx --model ckpt/yyy
  ```

* For molecule translation, replace `yyy` with your selected model in `ckpt/drd2`, `ckpt/logp04`, `ckpt/logp06` or `ckpt/qed`.

  ```bash
  python tools/translate.py --config configs/xxx --model ckpt/yyy
  ```

## Evaluation

Calculate metrics on testing result file and replace `zzz` with your result file in `results/*`.

```bash
python tools/eval.py --config configs/xxx --result results/zzz
```

