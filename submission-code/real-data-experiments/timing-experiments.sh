outdir=results/timing/

python train_bayes_sgd.py --device cuda:3 --dataset adult --batch-size 256 --epochs 10 --lr 0.15 --max-per-sample-grad_norm 1 --disable-dp --log $outdir/adult-disable-dp.jsonl
python train_bayes_sgd.py --device cuda:3 --dataset adult --batch-size 256 --epochs 10 --lr 0.15 --max-per-sample-grad_norm 1 --bayes-mia --log $outdir/adult-mia.jsonl
python train_bayes_sgd.py --device cuda:3 --dataset adult --batch-size 256 --epochs 10 --lr 0.15 --max-per-sample-grad_norm 1 --moments-accountant --log $outdir/adult-accountant.jsonl
python train_bayes_sgd.py --device cuda:3 --dataset adult --batch-size 256 --epochs 10 --lr 0.15 --max-per-sample-grad_norm 1 --bayes-ai-approximate --log $outdir/adult-ai-approximate.jsonl
python train_bayes_sgd.py --device cuda:3 --dataset adult --batch-size 256 --epochs 10 --lr 0.15 --max-per-sample-grad_norm 1 --bayes-ai  --log $outdir/adult-ai.jsonl