N_PROCS=5

seq 5 | parallel -P $N_PROCS python train_bayes_sgd.py --dataset adult --batch-size 256 --epochs 30 --lr 0.0001 --max-per-sample-grad_norm 2 --sigma 3.51 --delta 0.0000038 --bayes-mia --bayes-ai-approximate --moments-accountant --log results/analysis/adult-approximate-{}.jsonl