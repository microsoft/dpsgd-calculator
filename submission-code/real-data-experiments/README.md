# Experiments on the Adult and Purchase-100 datasets

This folder contains the code to run the experiments on the Adult and Purchase-100 datasets.

![image](https://github.com/microsoft/dpsgd-calculator/assets/2278118/38f4ac1c-f898-431f-a82f-2217147c516d)

All these experiments can be run via bash scripts (see below) to reproduce the exact same parameter sets that we used in our paper.

The first time these script run, they will download and pre-process the Adult and Purchase-100 datasets from HuggingFace and a GitHub repository from the Data Privacy and Trustworthy Machine Learning Research Lab at the National University of Singapore. You will need ~1.2GB free disk space for this.

 - https://huggingface.co/datasets/scikit-learn/adult-census-income
 - https://raw.githubusercontent.com/privacytrustlab/datasets/master/dataset_purchase.tgz

Results are plotted via the Jupyter notebook `plot_results.ipynb`.

Please ensure you have set up your environment as per [`submission-code`](/submission-code).


## Utility-privacy tradeoff analysis

To reproduce Figure 1:

```bash
bash launch-adult.sh
bash launch-purchase100.sh
```

This will create the log files `results/analysis/adult-approximate-{1..5}.jsonl` and `results/analysis/purchase100-{1..5}.jsonl`.

Note that the results for Adult are from the approximate AI analysis. Better bounds can be obtained via `launch-adult-full.sh`, but this takes much longer to run.

You can then parse and plot the results by running the appropriate section in the Jupyter notebook `plot_results.ipynb`.


## Running time comparison

This experiment reproduces Figure 8 in the paper, which compares the average running time per-epoch of DP-SGD, when instrumenting it to run the various analyses:

```bash
bash timing-experiments.sh
```

When done, plot the results evaluating the relevant cells in `plot_results.ipynb`.


## Full vs approximate AI analysis

To reproduce Figure 9:

```bash
bash launch-adult-full.sh
```

> [!NOTE]
> This script takes a long time to run (~25h on an A100), as it exactly computes the diamater of the set of per-sample gradients for all values of the sensitive attribute at each iteration.

This will create the log files `results/analysis/adult-full-{1..5}.jsonl`.

When done, plot the results evaluating the relevant cells in `plot_results.ipynb`.
