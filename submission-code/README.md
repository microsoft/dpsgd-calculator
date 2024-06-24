# Closed-Form Bounds for DP-SGD against Record-level Inference

This code accompanies the USENIX Security 2024 paper [Closed-Form Bounds for DP-SGD against Record-level Inference](https://www.usenix.org/conference/usenixsecurity24/presentation/cherubin). It allows reproducing all Figures and numerical results presented in the paper.


## Contents

The code is split into two parts:

- Part-A, in this directory, contains experiments to assess the quality of our theoretical bounds and compare them with prior work using numerical privacy accountants.

- Part-B, in the `real-data-experiments` subdirectory, contains experiments to evaluate our bounds on the Adult US Census Income dataset and Purchase-100, a dataset post-processded from Kaggle’s Acquire Valued Shoppers challenge.


```
.
├── README.md                 # This file.
|
├── evaluation.ipynb          # Jupyter notebook to reproduce Figs. 2-7, 10 and
|                             # numerical results in Sections 4 and 6.
|
├── libbayessgd.py            # Python module with functions to compute privacy
|                             # metrics with techniques from the paper and the
|                             ~ PLD accountant.
|
├── requirements.txt          # Python requirements.
|
└── real-data-experiments     # Part-B experiments.
    |
    ├── README.md             # High-level description of Part-B.
    |
    ├── bayessgd.py           # Opacus extensions to implement data-independent
    |                         # MI and data-dependent AI analysis.
    |
    ├── train_bayes_sgd.py    # Model training script from Opacus instrumented
    |                         # to compute MI and AI bounds.
    |
    ├── utils.py              # Dataset utilities and torch modules.
    |
    ├── launch-adult-full.sh  # Script to train MLPs on Adult and compute
    |                         # metrics, including exact AI bounds.
    |
    ├── launch-adult.sh       # Script to train MLPs on Adult and compute
    |                         # metrics, including approximate AI bounds.
    |
    ├── launch-purchase100.sh # Script to train ConvNets on Purchase-100 and
    |                         # compute metrics, including approx. AI bounds.
    |
    ├── timing-experiments.sh # Script to measure time required to run the
    |                         # various analyses.
    |
    └── plot_results.ipynb    # Jupyter notebook to plote the results
                              # obtained from running the bash scripts.
```

## Requirements

This code has been tested on Ubuntu 20.04 LTS and 22.04 LTS with Python 3.10. It should be possible to run it on other Linux distributions and other versions of Python >= 3.8 with no or minor adapatations.

The code for Part-B assumes a system with an NVIDIA GPU and CUDA. We have tested it on K80, V100, and A100 GPUs. It might be possible (but slow) to run it only on CPU by adding the `--device cpu` parameter to `train_bayes_sgd.py` in the various Bash scripts.

Besides the Python requirements listed in `requirements.txt`, some script use the GNU `parallel` utility. On Ubuntu, it could be installed with `sudo apt-get -y install parallel`.

## Running the code

We recommend using a Python virtual environment or Conda to avoid conflicts with any existing Python user or system-wide installations. For instance, to start from a fresh Conda environment you may use `conda create -n usenix python=3.10` followed by `conda activate usenix`.

Install all Python requirements with

```bash
$ pip install -r requirements.txt
```

### Part-A

For Part-A, start a Jupyter notebook server from a terminal by running

```bash
$ jupyter notebook
```

Once the Jupyter server starts, access it on a browser and open the `evaluation.ipynb` notebook. The notebook is self-documented. Evaluate the notebook cells sequentially to generate Figures 2-7 and 10 in the paper and the numerical results in Sections 4 and 6.


![image](https://github.com/microsoft/dpsgd-calculator/assets/2278118/f22f8cd2-9b7d-4e9c-a1ad-3f9492c75d9e)

## Part-B

If using an NVIDIA GPU, you may want to check that PyTorch is able to use it. You can do this by checking that the following Python program prints True:

```Python
import torch
print(torch.cuda.is_available())
```

The experiments on the `Adult` and `Purchase-100` datasets are implemented by instrumenting [`Opacus`](https://github.com/pytorch/opacus).
Please, see the `README.md` file in [`real-data-experiments`](/submission-code/real-data-experiments) for additional instructions.

## License

See [LICENSE.txt](..\LICENSE.txt).

## Citation

Please consider citing the paper if you found our work useful.

```
@InProceedings{closed_form:usenix2024,
  title      = {Closed-Form Bounds for {DP}-{SGD} against Record-level Inference},
  author     = {Cherubin, Giovanni and
                K{\"o}pf, Boris and
                Paverd, Andrew and
                Tople, Shruti and
                Wutschitz, Lukas and
                Zanella-B{\'e}guelin, Santiago},
  booktitle  = {33rd USENIX Security Symposium (USENIX Security 24)},
  year       = {2024},
  url        = {https://www.usenix.org/conference/usenixsecurity24/presentation/cherubin},
  publisher  = {USENIX Association}
}
```