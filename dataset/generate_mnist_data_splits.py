import json
import logging
import os

# add project root to sys path
import sys
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import set_logger


def generate_splits(
    data_path,
    save_path,
    name="mnist_splits.json",
    train_num=100000,
    test_num=100,
    digits=None,
):
    save_file = Path(save_path) / name
    inr_path = Path(data_path)
    data_split = defaultdict(lambda: defaultdict(list))

    digits = digits or range(10)  # If no digits specified, use all 0-9

    counts = {"train": defaultdict(int), "test": defaultdict(int)}

    max_counts = {"train": train_num, "test": test_num}

    for p in inr_path.glob("mnist_png_*/**/*.pth"):
        digit = int(p.parent.parent.stem.split("_")[-2])
        if digit not in digits:
            continue

        set_type = "train" if "training" in p.as_posix() else "test"

        if counts[set_type][digit] < max_counts[set_type]:
            counts[set_type][digit] += 1
        else:
            set_type = "val"
        data_split[set_type]["path"].append(p.resolve().as_posix())
        data_split[set_type]["label"].append(str(digit))

        if all(
            counts[t][d] >= max_counts[t] for t in ["train", "test"] for d in digits
        ):
            break

    logging.info(
        f"train size: {len(data_split['train']['path'])}, "
        f"val size: {len(data_split['val']['path'])}, "
        f"test size: {len(data_split['test']['path'])}"
    )

    with open(save_file, "w") as file:
        json.dump(data_split, file)

    logging.info(
        f"train size: {len(data_split['train']['path'])}, "
        f"val size: {len(data_split['val']['path'])}, test size: {len(data_split['test']['path'])}"
    )

    with open(save_file, "w") as file:
        json.dump(data_split, file)


if __name__ == "__main__":
    set_logger()
    parser = ArgumentParser("MNIST - generate data splits")
    parser.add_argument(
        "--data-path", type=str, default="./mnist-inrs", help="path to MNIST data"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=".",
        help="path to save MNIST data splits",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="mnist_splits.json",
        help="json file name",
    )

    args = parser.parse_args()

    digits = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    generate_splits(
        data_path=args.data_path,
        save_path=args.save_path,
        name=args.name,
        digits=digits,
        train_num=100000,
        test_num=100,
    )
