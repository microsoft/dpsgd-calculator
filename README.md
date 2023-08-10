# Closed-Form Bounds for DP-SGD against Record-level Inference

Welcome to the repository that accompanies the paper "Closed-Form Bounds for DP-SGD against Record-level Inference" by G. Cherubin, B. Köpf, A. Paverd, S. Tople, L. Wutschitz, and S. Zanella-Béguelin.

This repo contains:
- the code and notebooks to replicate our experiments
- an [interactive DP-SGD privacy calculator TODO: change link before deployment](https://stunning-adventure-z7gop8l.pages.github.io/) for membership inference.

# Replicating our experiments

Most figures in our paper are based on synthetic symulations. They can be reproduced by running the notebook `evaluation.ipynb`.

The experiments on the Adult and Purchase100 datasets are implemented by instrumenting `Opacus`.
They can be run by launching `launch_adult.sh` or `launch_purchase100.sh` from `submission-code/real-data-experiments`.
The results are parsed and plotted via the notebook `plot_results.ipynb`.