"""
Training script

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2024,
"""

import logging
import sys
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from cot.config import CHECK_DIR, TOKEN_DICT
from cot.data import BinaryCopy, MixedDataset, Parity, Polynomial
from cot.evals import EvaluationIO
from cot.evals.cot import AccuracyEval, FullEval
from cot.models import Transformer, TransformerConfig

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Reproducibility and Device
# -----------------------------------------------------------------------------

torch.manual_seed(100)
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.manual_seed_all(0)
else:
    device = torch.device("cpu")


def train(
    problem="polynomial",
    cot=True,
    data_dir=None,
    n_len=8,
    zipf_offset=0,
    zipf_coef=0,
    emb_dim=128,
    emb_dropout=0.1,
    pos_dim=None,
    freeze_pos=False,
    n_head=1,
    n_layer=2,
    n_epochs=1000,
    sgd=False,
    batch_size=None,
    learning_rate=1e-3,
    checkpoint_freq=100,
    overwrite_checkpoint=True,
    load_checkpoint=False,
    check_dir=None,
    full_eval=False,
    eval_freq=10,
):
    """
    Training a Transformer model on a specified problem.

    Paramters
    ---------
    problem: str
        Problem to be solved. Currently supported are "binary-copy", "parity", and "no-cot".
    cot: bool
        Wether to use chain-of-thought or not.
    data_dir: str
        Path to the directory where to save the data.
    n_len: int
        Maximum number of lenghts for sequences.
    zipf_offset: float
        Index offset to the Zipf law generating sentence lengths.
    zipf_coef: float
        Decaying coefficient to the Zipf law generating sentence lengths.
    emb_dim: int
        Embedding dimension size.
    emb_dropout: float
        Dropout rate for the embeddings.
    pos_dim: int
        Dimension of the positional embedding. Default is `emb_dim`.
    freeze_pos: bool
        Wether to learn the position embedding or to freeze them.
    n_head: int
        Number of attention heads.
    n_layer: int
        Number of layers.
    n_epochs: int
        Total number of training epochs.
    sgd: bool
        Wether to use SGD or Adam.
    batch_size: int
        Batch size. Default is full batch.
    learning_rate: float
        Learning rate.
    checkpoint_freq: int
        Checkpoint saving frequency.
    overwrite_checkpoint: bool
        Whether to overwrite existing checkpoints or not.
    load_checkpoint: bool
        Whether to load a previous checkpoint for continuing training.
    check_dir: str
        Path to checkpoint directory.
    full_eval: bool
        Wether to evaluate for the special circuit or not.
    eval_freq: int
        Evaluation frequency.
    """

    # -----------------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------------

    match problem:
        case "binary-copy":
            Problem = partial(BinaryCopy, cot=cot)
        case "parity":
            Problem = partial(Parity, cot=cot)
        case "polynomial":
            Problem = partial(Polynomial, cot=cot)
        case "mix":
            Problem = MixedDataset
        case _:
            raise ValueError(f"Problem {problem} not recognized.")

    # hyperparameters
    lengths = list(np.arange(n_len) + 1)

    trainset = Problem(save_dir=data_dir)
    trainset.set_data(lengths, data_type="train")

    testset = Problem(save_dir=data_dir)
    testset.set_data(lengths, data_type="test")

    if batch_size is None:
        batch_size = len(trainset)
        logger.info("No batch size specified. Using gradient descent (full batch).")

    # non-uniform sampler
    probas_by_len = (np.arange(len(lengths), dtype=float) + zipf_offset) ** (-zipf_coef)
    probas_by_len /= probas_by_len.sum()
    sampler = trainset.get_sampler_by_len(probas_by_len)

    loader = DataLoader(trainset, batch_size=batch_size, sampler=sampler)
    logger.info(f"Problem: {trainset.prefix}. Number of training data: {len(trainset)}.")

    # --------------------------------------------------------------------------
    # Model
    # --------------------------------------------------------------------------

    config = TransformerConfig(
        vocab_size=64,
        emb_dim=emb_dim,
        pos_emb=True,
        pos_dim=pos_dim,
        freeze_pos=freeze_pos,
        seq_len=len(trainset[0]),
        emb_dropout=emb_dropout,
        n_head=n_head,
        n_layer=n_layer,
    )

    losses = np.empty(n_epochs)

    if check_dir is None:
        check_dir = CHECK_DIR / trainset.prefix
    check_dir.mkdir(parents=True, exist_ok=True)

    model = Transformer(config)
    logger.debug(f"Model: {model}.")

    if sgd:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    logger.info(f"Device used: {device}.")
    model.to(device)
    probas_by_len = torch.from_numpy(probas_by_len).to(device=device)

    if load_checkpoint:
        path = check_dir / "model.pth"
        logger.info(f"Loading from checkpoint {path}.")
        checkpoint = torch.load(path)

        epoch = checkpoint["epoch"]

        if epoch > n_epochs:
            logger.error(f"Model has been trained for {epoch} epochs, which is higher than {n_epochs}")
            sys.exit()
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        losses[:epoch] = checkpoint["losses"][:epoch]
    else:
        epoch = 0

    # --------------------------------------------------------------------------
    # Evaluation Placeholders
    # --------------------------------------------------------------------------

    if full_eval:
        evaluator = FullEval(lengths)
    else:
        evaluator = AccuracyEval(lengths)
    eval_dim = evaluator.eval_dim

    def eval(model):
        with torch.no_grad():
            model.eval()
            train_evals = evaluator(model, trainset)
            test_evals = evaluator(model, testset)
        model.train()
        return torch.hstack((train_evals, test_evals))

    if load_checkpoint:
        n_evals = (n_epochs - epoch) // eval_freq + 1
        report_eval = EvaluationIO(
            n_evals, 2 * eval_dim, past_evals=checkpoint["evals"], past_timestamps=checkpoint["timestamps"]
        )
    else:
        n_evals = n_epochs // eval_freq + 1
        report_eval = EvaluationIO(n_evals, 2 * eval_dim)
        evals = eval(model)
        report_eval(epoch, evals)

    # --------------------------------------------------------------------------
    # Training loop
    # --------------------------------------------------------------------------

    logger.info(f"Starting Training from epoch {epoch}.")
    model.train()
    while epoch < n_epochs:
        # training
        running_loss = 0
        for sequence in loader:
            sequence = sequence.to(device=device, dtype=torch.long)

            inputs = sequence[:, :-1]
            targets = sequence[:, 1:]

            # only train on the chain-of-thoughts process, EoI is represented by 1 in our case
            ind = targets == TOKEN_DICT["EoI"]
            cot_mask = ind.cumsum(axis=1)
            cot_mask[ind] = 0
            cot_mask = cot_mask.to(dtype=bool)

            logits = model(inputs)
            loss = F.cross_entropy(logits[cot_mask].view(-1, logits.size(-1)), targets[cot_mask].reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                running_loss += loss.item()

        losses[epoch] = running_loss
        epoch = epoch + 1
        logger.info(f"Epoch {epoch:5d}, Loss: {running_loss:.4f}")

        # evaluation
        if not epoch % eval_freq:
            evals = eval(model)
            report_eval(epoch, evals)

            accuracy = (evals[0 : len(lengths)] * probas_by_len).sum().item()
            test_accuracy = (evals[eval_dim : eval_dim + len(lengths)] * probas_by_len).sum().item()
            logger.info(f"Epoch {epoch:5d}, Accuracy: {accuracy:.4f}, {test_accuracy:.4f}")

        # checkpointing
        if not epoch % checkpoint_freq or epoch == n_epochs:
            if overwrite_checkpoint:
                path = check_dir / "model.pth"
            else:
                path = check_dir / f"model_{epoch}.pth"

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "losses": losses,
                    "evals": report_eval.evals,
                    "timestamps": report_eval.timestamps,
                    "meaning": evaluator.meaning,
                },
                path,
            )
            logger.info(f"Checkpointing model at {path}.")
    logger.info("Training finished.")


if __name__ == "__main__":
    import fire

    from cot.config import logging_datefmt, logging_format, logging_level

    logging.basicConfig(
        format=logging_format,
        datefmt=logging_datefmt,
        style="{",
        level=logging_level,
        handlers=[logging.StreamHandler()],
    )

    fire.Fire(train)
