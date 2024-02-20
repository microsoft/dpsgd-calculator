# Replicating our experiments

Please, install all the requirements via:

```bash
$ pip install -r requirements.txt
```

Most figures in our paper are based on synthetic symulations. They can be reproduced by running the notebook [`evaluation.ipynb`](submission-code/evaluation.ipynb).
You will need to have Jupyter installed.

![image](https://github.com/microsoft/dpsgd-calculator/assets/2278118/f22f8cd2-9b7d-4e9c-a1ad-3f9492c75d9e)


The experiments on the `Adult` and `Purchase100` datasets are implemented by instrumenting [`Opacus`](https://github.com/pytorch/opacus).
Please, head to [`real-data-experiments`](/submission-code/real-data-experiments) for instructions.
