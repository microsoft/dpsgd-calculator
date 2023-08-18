#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
from opacus import PrivacyEngine
from tqdm import tqdm

from utils import *
from bayessgd import AIAnalysis, MIAAnalysis, UserMIAAnalysis


def train(args, model, device, train_loader, optimizer, privacy_engine, epoch, test_loader):
    start_time = time.time()
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)

        if args.bayes_mia:
            privacy_engine.bayes_mia_analysis.step()
        if args.bayes_ai:
            privacy_engine.bayes_ai_analysis.step(model, data, target, optimizer, criterion, device)
        if args.bayes_ai_approximate:
            privacy_engine.bayes_ai_analysis_approximate.step(model, data, target, optimizer, criterion, device)
        if args.bayes_user_mia:
            privacy_engine.bayes_user_mia_analysis.step()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    if args.log:
        log = {
        "epoch": epoch,
        "loss": np.mean(losses),
        "epoch_time": time.time() - start_time,
        "n_steps": len(train_loader),
        "accuracy": test(model, device, test_loader),
        }

        if not args.disable_dp:
            if args.bayes_mia:
                log["Bayes-MIA"] = privacy_engine.bayes_mia_analysis.beta.item()
            if args.bayes_ai:
                log["Bayes-AI"] = privacy_engine.bayes_ai_analysis.beta.item()
            if args.bayes_ai_approximate:
                log["Bayes-AI-approximate"] = privacy_engine.bayes_ai_analysis_approximate.beta.item()
            if args.bayes_user_mia:
                log["Bayes-user-MIA"] = privacy_engine.bayes_user_mia_analysis.beta.item()
            if args.moments_accountant:
                log["epsilon"] = privacy_engine.accountant.get_epsilon(delta=args.delta)
                log["delta"] = args.delta

        with open(args.log, "a") as f:
            f.write(json.dumps(log) + "\n")


def test(model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return correct / len(test_loader.dataset)


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description="Bayes SGD analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=64,
        metavar="B",
        help="Batch size",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1024,
        metavar="TB",
        help="input batch size for testing",
    )
    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train",
    )
    parser.add_argument(
        "-r",
        "--n-runs",
        type=int,
        default=1,
        metavar="R",
        help="number of runs to average on",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        metavar="S",
        help="Noise multiplier",
    )
    parser.add_argument(
        "-c",
        "--max-per-sample-grad_norm",
        type=float,
        default=1.0,
        metavar="C",
        help="Clip per-sample gradients to this norm",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target delta",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="GPU ID for this process",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="Save the trained model",
    )
    parser.add_argument(
        "--disable-dp",
        action="store_true",
        default=False,
        help="Disable privacy training and just train with vanilla SGD",
    )
    parser.add_argument(
        "--bayes-mia",
        action="store_true",
        default=False,
        help="Enable the Bayes-SGD MIA analysis",
    )
    parser.add_argument(
        "--bayes-ai",
        action="store_true",
        default=False,
        help="Enable the Bayes-SGD AI analysis",
    )
    parser.add_argument(
        "--bayes-ai-approximate",
        action="store_true",
        default=False,
        help="Enable the approximate Bayes-SGD AI analysis",
    )
    parser.add_argument(
        "--bayes-user-mia",
        type=int,
        default=0,
        help="Enable user-level MIA analysis for a user whose data has the specified size. E.g., `--bayes-user-mia 100`.",
    )
    parser.add_argument(
        "--moments-accountant",
        action="store_true",
        default=False,
        help="Enable DP accountant",
    )
    parser.add_argument(
        "--secure-rng",
        action="store_true",
        default=False,
        help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="Where data is/will be stored",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        help="Dataset/task: [mnist,purchase100,adult]",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Logging file (format: jsonl)",
    )
    args = parser.parse_args()
    #import random
    #args.device = random.choice(["cuda:0", "cuda:1", "cuda:2", "cuda:3"])

    device = torch.device(args.device)

    if args.dataset == "mnist":
        loader = mnist
        attribute_idx = MNIST_ATTRIBUTE_IDX
        attribute_range = MNIST_ATTRIBUTE_RANGE
    elif args.dataset == "purchase100":
        loader = purchase100
        attribute_idx = PURCHASE100_ATTRIBUTE_IDX
        attribute_range = PURCHASE100_ATTRIBUTE_RANGE
    elif args.dataset == "adult":
        loader = adult
        attribute_idx = ADULT_ATTRIBUTE_IDX
        attribute_range = ADULT_ATTRIBUTE_RANGE
    else:
        raise NotImplementedError

    train_loader, test_loader, init_model, init_optimizer = loader(lr=args.lr, batch_size=args.batch_size,
                                                            test_batch_size=args.test_batch_size,
                                                            data_root=args.data_root)

    run_results = []
    for _ in range(args.n_runs):
        model = init_model().to(device)
        optimizer = init_optimizer(model)
        privacy_engine = None

        if not args.disable_dp:
            privacy_engine = PrivacyEngine(secure_mode=args.secure_rng)
            model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=args.sigma,
                max_grad_norm=args.max_per_sample_grad_norm,
            )

            if not args.moments_accountant:
                optimizer.attach_step_hook(lambda _: None)

        # Initialize the Bayes-SGD privacy analysis.
        if args.bayes_mia:
            privacy_engine.bayes_mia_analysis = MIAAnalysis(
                noise_multiplier=optimizer.noise_multiplier,
                sample_rate=train_loader.sample_rate,
            )

        if args.bayes_user_mia > 0:
            privacy_engine.bayes_user_mia_analysis = UserMIAAnalysis(
                noise_multiplier=optimizer.noise_multiplier,
                sample_rate=train_loader.sample_rate,
                user_data_size=args.bayes_user_mia,
            )

        # NOTE: we keep approximate and "full" AI analysis separate for
        # the paper experiments.
        if args.bayes_ai:
            privacy_engine.bayes_ai_analysis = AIAnalysis(
                attribute_idx=attribute_idx,
                attribute_range=attribute_range,
                sample_rate=train_loader.sample_rate,
                max_grad_norm=optimizer.max_grad_norm,
                noise_multiplier=optimizer.noise_multiplier,
                approximate=False,
            )

        if args.bayes_ai_approximate:
            privacy_engine.bayes_ai_analysis_approximate = AIAnalysis(
                attribute_idx=attribute_idx,
                attribute_range=attribute_range,
                sample_rate=train_loader.sample_rate,
                max_grad_norm=optimizer.max_grad_norm,
                noise_multiplier=optimizer.noise_multiplier,
                approximate=True,
            )

        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, privacy_engine, epoch, test_loader)
        run_results.append(test(model, device, test_loader))

    if len(run_results) > 1:
        print(
            "Accuracy averaged over {} runs: {:.2f}% ± {:.2f}%".format(
                len(run_results), np.mean(run_results) * 100, np.std(run_results) * 100
            )
        )

    repro_str = (
        f"mnist_{args.lr}_{args.sigma}_"
        f"{args.max_per_sample_grad_norm}_{args.batch_size}_{args.epochs}"
    )
    torch.save(run_results, f"run_results_{repro_str}.pt")

    if args.save_model:
        torch.save(model.state_dict(), f"mnist_cnn_{repro_str}.pt")


if __name__ == "__main__":
    main()
