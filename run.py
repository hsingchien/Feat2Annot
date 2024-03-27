#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2020-21: Homework 4
run.py: Run Script for Simple NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Vera Lin <veralin@stanford.edu>

Usage:
    run.py train --path=<file> --window-size=<int> --feature-num=<int> [options]
    run.py decode [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    run.py decode [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --seed=<int>                            seed [default: 0]
    --path=<str>                            path to dataset [default: "./data"]
    --batch-size=<int>                      batch size [default: 32]
    --train-size=<float>                    training set proportion [default: 0.95]
    --feature-num=<int>                     feature num [default: 100]
    --window-size=<int>                     window size of encoder/decoder [default: 15]
    --hidden-size=<int>                     hidden size [default: 1024]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 30]
    --input-feed                            use input feeding
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 5]
    --sample-size=<int>                     sample size [default: 5]
    --lr=<float>                            learning rate [default: 0.001]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path [default: model.bin]
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --dropout=<float>                       dropout [default: 0.3]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
"""
import math
import sys
import pickle
import time
import os

directory_separator = os.sep


# from docopt import docopt
from Feat2Annot import Hypothesis, Feat2AnnotModel
from util import PoseDataset, prepare_dataset
import numpy as np
from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm
import torch
import torch.nn.utils
from torch.utils.data import DataLoader
from torcheval import metrics
from docopt import docopt


def train(args: Dict):
    """Train the Pose-Annotation Model.
    @param args (Dict): args from cmd line
    """
    device = torch.device("cuda:0" if args["--cuda"] else "cpu")
    # Prepare training data and validation data

    dataset = prepare_dataset(
        str(args["--path"]), int(args["--window-size"]), device=device
    )
    train_proportion = float(args["--train-size"])
    train_batch_size = int(args["--batch-size"])
    clip_grad = float(args["--clip-grad"])
    valid_niter = int(args["--valid-niter"])
    log_every = int(args["--log-every"])
    model_save_path = args["--save-to"]

    generator1 = torch.Generator().manual_seed(42)
    train_data, val_data = torch.utils.data.random_split(
        dataset, (train_proportion, 1 - train_proportion), generator1
    )
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
    val_dataloader = DataLoader(val_data, shuffle=False, batch_size=1)

    model = Feat2AnnotModel(
        input_size=int(args["--feature-num"]),
        hidden_size=int(args["--hidden-size"]),
        dropout_rate=float(args["--dropout"]),
        target_class=dataset.get_annot_class(),
    )
    model = model.to(device)
    model.train()

    uniform_init = float(args["--uniform-init"])
    if np.abs(uniform_init) > 0.0:
        print(
            "uniformly initialize parameters [-%f, +%f]" % (uniform_init, uniform_init),
            file=sys.stderr,
        )
        for p in model.parameters():
            p.data.uniform_(-uniform_init, uniform_init)

    print("use device: %s" % device, file=sys.stderr)

    metric = metrics.MulticlassAccuracy()

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args["--lr"]))

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cum_tgt_words = (
        report_tgt_words
    ) = 0
    cum_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print("begin Maximum Likelihood training")

    while True:
        epoch += 1

        for source_feature, tgt_annot in train_dataloader:
            train_iter += 1
            model.train()

            optimizer.zero_grad()

            example_losses = -model(source_feature, tgt_annot)  # (batch_size,)
            batch_loss = example_losses.sum()

            loss = batch_loss / train_batch_size

            loss.backward()

            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()

            batch_losses_val = batch_loss.item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val

            report_examples += train_batch_size
            cum_examples += train_batch_size

            if train_iter % log_every == 0:
                print(
                    "epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f "
                    "cum. examples %d, speed %.2f seqs/sec, time elapsed %.2f sec"
                    % (
                        epoch,
                        train_iter,
                        report_loss / report_examples,
                        math.exp(report_loss / report_examples),
                        cum_examples,
                        cum_examples / (time.time() - train_time),
                        time.time() - begin_time,
                    ),
                    file=sys.stderr,
                )

                train_time = time.time()
                report_loss = report_examples = 0.0

            # perform validation
            if train_iter % valid_niter == 0:
                print(
                    "epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d"
                    % (
                        epoch,
                        train_iter,
                        cum_loss / cum_examples,
                        np.exp(cum_loss / cum_examples),
                        cum_examples,
                    ),
                    file=sys.stderr,
                )

                cum_loss = cum_examples = 0.0
                valid_num += 1

                print("begin validation ...", file=sys.stderr)

                # compute validation score
                model.eval()
                metric.reset()
                for source_feature, tgt_annot in tqdm(val_dataloader):
                    annot_hypothesis = model.beam_search(source_feature, 1)
                    annot_hat = torch.tensor(
                        annot_hypothesis[0].value, dtype=torch.int64
                    )
                    metric.update(annot_hat, tgt_annot.squeeze(0))
                valid_metric = metric.compute()
                print(
                    "validation: iter %d, multiclass accuracy %f"
                    % (train_iter, valid_metric),
                    file=sys.stderr,
                )

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(
                    hist_valid_scores
                )
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print(
                        "save currently the best model to [%s]" % model_save_path,
                        file=sys.stderr,
                    )
                    model.save(model_save_path)

                    # also save the optimizers' state
                    torch.save(optimizer.state_dict(), model_save_path + ".optim")
                elif patience < int(args["--patience"]):
                    patience += 1
                    print("hit patience %d" % patience, file=sys.stderr)

                    if patience == int(args["--patience"]):
                        num_trial += 1
                        print("hit #%d trial" % num_trial, file=sys.stderr)
                        if num_trial == int(args["--max-num-trial"]):
                            print("early stop!", file=sys.stderr)
                            exit(0)

                        # decay lr, and restore from previously best checkpoint
                        lr = optimizer.param_groups[0]["lr"] * float(args["--lr-decay"])
                        print(
                            "load previously best model and decay learning rate to %f"
                            % lr,
                            file=sys.stderr,
                        )

                        # load model
                        params = torch.load(
                            model_save_path, map_location=lambda storage, loc: storage
                        )
                        model.load_state_dict(params["state_dict"])
                        # model = model.to(device)

                        print("restore parameters of the optimizers", file=sys.stderr)
                        optimizer.load_state_dict(
                            torch.load(model_save_path + ".optim")
                        )

                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = lr

                        # reset patience
                        patience = 0

                if epoch == int(args["--max-epoch"]):
                    print("reached maximum number of epochs!", file=sys.stderr)
                    exit(0)


# def decode(args: Dict[str, str]):
#     """Performs decoding on a test set, and save the best-scoring decoding results.
#     If the target gold-standard sentences are given, the function also computes
#     corpus-level BLEU score.
#     @param args (Dict): args from cmd line
#     """

#     print(
#         "load test source sentences from [{}]".format(args["TEST_SOURCE_FILE"]),
#         file=sys.stderr,
#     )
#     test_data_src = read_corpus(args["TEST_SOURCE_FILE"], source="src", vocab_size=3000)
#     if args["TEST_TARGET_FILE"]:
#         print(
#             "load test target sentences from [{}]".format(args["TEST_TARGET_FILE"]),
#             file=sys.stderr,
#         )
#         test_data_tgt = read_corpus(
#             args["TEST_TARGET_FILE"], source="tgt", vocab_size=2000
#         )

#     print("load model from {}".format(args["MODEL_PATH"]), file=sys.stderr)
#     model = NMT.load(args["MODEL_PATH"])

#     if args["--cuda"]:
#         model = model.to(torch.device("cuda:0"))

#     hypotheses = beam_search(
#         model,
#         test_data_src,
#         #  beam_size=int(args['--beam-size']),
#         beam_size=10,
#         max_decoding_time_step=int(args["--max-decoding-time-step"]),
#     )

#     if args["TEST_TARGET_FILE"]:
#         top_hypotheses = [hyps[0] for hyps in hypotheses]
#         bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
#         print("Corpus BLEU: {}".format(bleu_score), file=sys.stderr)

#     with open(args["OUTPUT_FILE"], "w") as f:
#         for src_sent, hyps in zip(test_data_src, hypotheses):
#             top_hyp = hyps[0]
#             hyp_sent = "".join(top_hyp.value).replace("▁", " ")
#             f.write(hyp_sent + "\n")


# def beam_search(
#     model: NMT,
#     test_data_src: List[List[str]],
#     beam_size: int,
#     max_decoding_time_step: int,
# ) -> List[List[Hypothesis]]:
#     """Run beam search to construct hypotheses for a list of src-language sentences.
#     @param model (NMT): NMT Model
#     @param test_data_src (List[List[str]]): List of sentences (words) in source language, from test set.
#     @param beam_size (int): beam_size (# of hypotheses to hold for a translation at every step)
#     @param max_decoding_time_step (int): maximum sentence length that Beam search can produce
#     @returns hypotheses (List[List[Hypothesis]]): List of Hypothesis translations for every source sentence.
#     """
#     was_training = model.training
#     model.eval()

#     hypotheses = []
#     with torch.no_grad():
#         for src_sent in tqdm(test_data_src, desc="Decoding", file=sys.stdout):
#             example_hyps = model.beam_search(
#                 src_sent,
#                 beam_size=beam_size,
#                 max_decoding_time_step=max_decoding_time_step,
#             )

#             hypotheses.append(example_hyps)

#     if was_training:
#         model.train(was_training)

#     return hypotheses


def main():
    """Main func."""
    args = docopt(__doc__)

    # Check pytorch version
    assert (
        torch.__version__ >= "1.0.0"
    ), "Please update your installation of PyTorch. You have {} and you should have version 1.0.0".format(
        torch.__version__
    )

    # seed the random number generators
    seed = int(args["--seed"])
    torch.manual_seed(seed)
    if args["--cuda"]:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    if args["train"]:
        train(args)
    # elif args["decode"]:
    #     decode(args)
    else:
        raise RuntimeError("invalid run mode")


if __name__ == "__main__":
    main()
