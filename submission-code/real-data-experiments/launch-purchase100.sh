N_PROCS=5
outdir=results/analysis/

mkdir -p $outdir

seq 5 | parallel -P $N_PROCS python train_bayes_sgd.py --dataset purchase100 --batch-size 512 --epochs 30 --lr 0.001 --max-per-sample-grad_norm 2.6 --sigma 1.8 --delta 0.0000004 --bayes-mia --bayes-ai --moments-accountant --log $outdir/purchase100-{}.jsonl