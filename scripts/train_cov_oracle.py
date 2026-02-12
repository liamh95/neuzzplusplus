#!/usr/bin/env python
# Copyright (c) 2023 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
Script communicating with the AFL++ custom mutator for Neuzz++ neural program smoothing.

- Train model for predicting edge or memory coverage based on seed content.
- Generate gradient information for each seed requested by AFL++ custom mutator.
- Communicate with custom mutator via named pipes.
"""
import argparse
import logging
import os
import pathlib
import sys
import time
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import random_split, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, average_precision_score

from neuzzpp.data_loaders import SeedFolderDataset, CoverageSeedDataset
from neuzzpp.models import MLP
from neuzzpp.mutations import compute_one_mutation_info
from neuzzpp.utils import (create_work_folders,
                           model_needs_retraining, EarlyStopping)

# Configure logger - console
logger = logging.getLogger("neuzzpp")
logger.setLevel(logging.INFO)
console_logger = logging.StreamHandler(sys.stdout)
log_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_logger.setFormatter(log_formatter)
logger.addHandler(console_logger)

# Use GPU if we can
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: Tensorboard stuff?



def train_model(args: argparse.Namespace, seed_dataset: SeedFolderDataset) -> MLP:
    """
    Function loading the dataset from the seeds folder, building and training the model.

    Args:
        args: Input args of the script.
        seed_dataset: Dataset initialized for the seed folder that will be used
            for training.
   Returns:
        Trained model.
    """ 
    # Load and split dataset, setup dataloaders
    seed_dataset.load_seeds_from_folder()
    rng = torch.Generator().manual_seed(args.random_seed) if args.random_seed is not None else None
    if args.val_split > 0.0:
        training_dataset, validation_dataset = random_split(seed_dataset, [1.0 - args.val_split, args.val_split], generator=rng)
        train_dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        training_dataset = seed_dataset
        # Set this to training_dataset?
        validation_dataset = None
        train_dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataloader = None

    # Compute class frequencies and weights
    _, initial_bias = seed_dataset.get_class_weights()

    # Create training callbacks
    seeds_path = pathlib.Path(args.seeds)
    model_path = seeds_path.parent / "models"

    # Set up model
    model = MLP(
        input_dim=seed_dataset.max_file_size,
        output_dim=seed_dataset.max_bitmap_size,
        learning_rate=args.lr,
        hidden_dim=args.n_hidden_neurons,
        output_bias=initial_bias,
        fast=args.fast,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1000)
    criterion = torch.nn.BCEWithLogitsLoss()
    early_stopping = EarlyStopping()

    for epoch in range(args.epochs):
        train_loss = 0.0
        val_loss = 0.0
        total_samples = 0

        # training
        model.train()
        for batch_inputs, batch_labels in train_dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(batch_inputs)
            loss = criterion(logits, batch_labels.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_inputs.size(0)
            scheduler.step()
        
        # validation
        model.eval()
        with torch.no_grad():
            if val_dataloader is not None:
                for batch_inputs, batch_labels in val_dataloader:
                    batch_inputs = batch_inputs.to(device)
                    batch_labels = batch_labels.to(device)

                    logits = model(batch_inputs)
                    loss = criterion(logits, batch_labels.float())
                    val_loss += loss.item() * batch_inputs.size(0)
                    total_samples += batch_inputs.size(0)
            else:
                for batch_inputs, batch_labels in train_dataloader:
                    logits = model(batch_inputs)
                    loss = criterion(logits, batch_labels.float())
                    val_loss += loss.item() * batch_inputs.size(0)
                    total_samples += batch_inputs.size(0)
        avg_val_loss = val_loss / total_samples if total_samples > 0 else 0
            
        # also saves best model (just state_dict to memory, not disk)
        early_stopping(avg_val_loss, model)

        if early_stopping.early_stop:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    # checkpointing
    if early_stopping.best_model_state is not None:
        model.load_state_dict(early_stopping.best_model_state)
        if not args.fast:
            best_model_path = model_path / "model_best.pt"
            torch.save(model.state_dict(), best_model_path)




    # Compute evaluation metrics on validation data
    if args.val_split > 0.0:
        model.eval()
        class_threshold = 0.5  # Classification threshold, use default of 0.5
        all_preds = []
        all_labels = []
        with torch.no_grad():
            assert val_dataloader is not None
            for batch_inputs, batch_labels in val_dataloader:
                logits = model(batch_inputs)
                probs = torch.sigmoid(logits)
                all_preds.append(probs.numpy())
                all_labels.append(batch_labels.numpy())        

        all_preds = torch.cat(all_preds, dim=0).numpy()
        y_true = torch.cat(all_labels, dim=0).numpy()
    
        y_pred = all_preds > class_threshold
        acc = accuracy_score(y_true.flatten(), y_pred.flatten())

        tp = np.sum((y_true == y_pred) & (y_pred == 1), axis=0)
        tn = np.sum((y_true == y_pred) & (y_pred == 0), axis=0)
        fp = np.sum((y_true != y_pred) & (y_pred == 1), axis=0)
        fn = np.sum((y_true != y_pred) & (y_pred == 0), axis=0)

        assert (tp + tn + fp + fn == y_true.shape[0]).all()
        assert tp.shape[0] == y_true.shape[1]

        prec = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=np.float64), where=(tp + fp) != 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=np.float64), where=(tp + fn) != 0)
        f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp, dtype=np.float64), where=(2*tp + fp + fn) != 0)
        pr_auc = average_precision_score(y_true, all_preds, average="micro")
        print(
            f"Acc: {acc}, prec: {prec.mean()}, recall: {recall.mean()}, f1: {f1.mean()}, pr-auc: {pr_auc}"
        )
            
    return model




def create_parser() -> argparse.ArgumentParser:
    """
    Create and return the parser instance for the CLI arguments.

    Returns:
        Parser object.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-l", "--max_len", help="maximum seed length", type=int, default=None)
    parser.add_argument(
        "-p",
        "--percentile_len",
        help="percentile of seed length to keep as maximum seed length (1-100); "
        "ignored if max_len is provided",
        type=int,
        default=80,
    )
    parser.add_argument(
        "-e", "--epochs", help="number of epochs for model training", type=int, default=100
    )
    parser.add_argument(
        "-b", "--batch_size", help="batch size for model training", type=int, default=32
    )
    parser.add_argument("--lr", help="learning rate", type=float, default=1e-4)
    parser.add_argument("-s", "--early_stopping", help="early stopping patience", type=int)
    parser.add_argument(
        "--n_hidden_neurons", help="number of neurons in hidden layer", type=int, default=4096
    )
    parser.add_argument(
        "-v",
        "--val_split",
        help="amount of data between 0 and 1 to reserve for validation",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "-r", "--random_seed", help="seed for random number generator", type=int, default=None
    )
    parser.add_argument(
        "-f",
        "--fast",
        help="train faster by skipping detailed model evaluation",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-c",
        "--cov",
        help="type of coverage to measure",
        type=str,
        choices=["edge"],  # Only edge coverage supported for now
        default="edge",
    )
    parser.add_argument("input_pipe", help="", type=str)
    parser.add_argument("output_pipe", help="", type=str)
    parser.add_argument(
        "seeds", help="path to seeds folder, usually named `queue` for AFL++", type=str
    )
    parser.add_argument(
        "target",
        help="target program and arguments",
        type=str,
        nargs=argparse.REMAINDER,
        metavar="target [target_args]",
    )
    return parser


def main(argv: Sequence[str] = tuple(sys.argv)) -> None:
    n_seeds_last_training: int = 0
    time_last_training: int = 0

    parser = create_parser()
    args = parser.parse_args(argv[1:])

    # Configure logger - file
    seeds_path = pathlib.Path(args.seeds)
    file_logger = logging.FileHandler(seeds_path.parent / "training.log")
    file_logger.setFormatter(log_formatter)
    logger.addHandler(file_logger)

    # Check that input and output named pipes exist
    input_pipe = pathlib.Path(args.input_pipe)
    output_pipe = pathlib.Path(args.output_pipe)
    if not input_pipe.is_fifo() or not output_pipe.is_fifo():
        raise ValueError("Input or output pipes do not exist.")

    # Validate inputs
    if args.percentile_len <= 0 or args.percentile_len > 100:
        raise ValueError(
            f"Invalid `percentile_len`. "
            f"Expected integer in [1, 100], received: {args.percentile_len}."
        )
    if args.val_split < 0.0 or args.val_split >= 1.0:
        raise ValueError(
            f"Invalid `val_split`. Expected value in [0, 1), received: {args.val_split}."
        )
    create_work_folders(seeds_path.parent)

    data_loader = CoverageSeedDataset(  # Only edge coverage supported for now
        seeds_path,
        args.target,
        args.max_len,
        args.percentile_len
    )
    model: Optional[MLP] = None
    out_pipe = open(output_pipe, "w")
    max_grads = os.environ.get("NEUZZPP_MAX_GRADS")
    n_grads = None if max_grads is None else int(max_grads)
    with open(input_pipe, "r") as seed_fifo:
        for seed_name in seed_fifo:
            # (Re-)train model if necessary
            if (
                model_needs_retraining(seeds_path, time_last_training, n_seeds_last_training)
                or model is None
            ):
                # Update info for model retraining
                n_seeds_last_training = len(list(seeds_path.glob("id*")))
                time_last_training = int(time.time())

                model = train_model(args, data_loader) # now our model outputs logits
                model.to(device)
                # model = create_logits_model(model)

            # Generate gradients for requested seed
            target_path = seeds_path / seed_name.strip()
            sorting_index_lst, gradient_lst = compute_one_mutation_info(
                model, target_path, n_grads
            )
            out_pipe.write(",".join(sorting_index_lst) + "|" + ",".join(gradient_lst) + "\n")
            out_pipe.flush()
    out_pipe.close()


if __name__ == "__main__":
    main()
