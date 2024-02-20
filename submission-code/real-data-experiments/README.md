# Experiments on the Adult and Purchase100 datasets

This folder contains the code to run the experiments on the Adult and Purchase100 datasets.
We used an Azure Standard NC24 (4 x NVIDIA Tesla K80 GPUs) instance.

![image](https://github.com/microsoft/dpsgd-calculator/assets/2278118/38f4ac1c-f898-431f-a82f-2217147c516d)

All these experiments can be run via bash scripts (see below) to reproduce the exact same parameter sets
that we used in our paper.
Results are plotted via the Jupyter notebook `plot_results.ipynb`.
Please, install the Python requirements as per [`submission-code`](/submission-code).

## Utility-privacy analysis

To reproduce Figure 1:

```bash
bash launch-adult.sh
bash launch-purchase100.sh
```

This will create the log files `results/analysis/adult-approximate-{1..5}.jsonl` and `results/analysis/purchase100-{1..5}.jsonl`.
Note that the results for Adult are from the approximate AI analysis. Better bounds can be obtained via `launch-adult-full.sh`, but this takes much longer to run.

You can then parse and plot the results by running the appropriate section of the jupyter notebook `plot_results.ipynb`.

## Full vs approximate AI analysis

To reproduce Figure 9:

```bash
bash launch-adult-full.sh
```

This will create the log files `results/analysis/adult-full-{1..5}.jsonl`.

They can be parsed and plotted by using `plot_results.ipynb`.

## Running time

This experiment (Figure 8) compares the running time of the various methods when running on top of DP-SGD.
To run them:

```bash
bash timing-experiments.sh
```

Then parse and plot the results via `plot_results.ipynb`.
