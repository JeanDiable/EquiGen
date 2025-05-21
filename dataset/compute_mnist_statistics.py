import os
import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from data.inr_dataset import Batch, INRDataset


def compute_stats(data_path: str, save_path: str, batch_size: int = 100000):
    train_set = INRDataset(path=data_path, split="train", normalize=False)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=8
    )

    batch: Batch = next(iter(train_loader))[0]
    weights_mean = [w.mean(0) for w in batch.weights]
    weights_std = [w.std(0) for w in batch.weights]
    biases_mean = [w.mean(0) for w in batch.biases]
    biases_std = [w.std(0) for w in batch.biases]
    print("weights_mean", weights_mean)
    print("weights_mean shape", weights_mean[0].shape)
    print("weights_std", weights_std)
    print("weights_std shape", len(weights_std))
    print("biases_mean", biases_mean)
    print("biases_mean shape", len(biases_mean))
    print("biases_std", biases_std)
    print("biases_std shape", len(biases_std))

    statistics = {
        "weights": {"mean": weights_mean, "std": weights_std},
        "biases": {"mean": biases_mean, "std": biases_std},
    }

    out_path = Path(save_path)
    out_path.mkdir(exist_ok=True, parents=True)
    torch.save(statistics, out_path / "mnist_statistics.pth")


if __name__ == "__main__":
    parser = ArgumentParser("MNIST - generate statistics")
    parser.add_argument("--data_path", type=str, default="mnist_splits.json")
    parser.add_argument("--save_path", type=str, default=".")
    parser.set_defaults(batch_size=100000, save_path=".")
    args = parser.parse_args()

    compute_stats(
        data_path=args.data_path, save_path=args.save_path, batch_size=args.batch_size
    )
