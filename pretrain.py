import logging
from argparse import ArgumentParser

import torch
import wandb
from torch import nn
from tqdm import trange

from data.inr_dataset import INRDataset
from models.nn.models import DWSModelForClassification
from utils.helpers import count_parameters, get_device, set_seed, str2bool
from utils.info_cse_loss import info_nce_loss


def main(
    path,
    epochs: int,
    lr: float,
    batch_size: int,
    device,
):
    # load dataset
    train_set = INRDataset(
        path=path,
        split="train",
        normalize=args.normalize,
        augmentation=args.augmentation,
        permutation=args.permutation,
        smooth=args.smooth,
        statistics_path=args.statistics_path,
    )
    val_set = INRDataset(
        path=path,
        split="test",
        normalize=args.normalize,
        augmentation=args.augmentation,
        permutation=args.permutation,
        smooth=args.smooth,
        statistics_path=args.statistics_path,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        shuffle=False,
    )

    logging.info(f"train size {len(train_set)}, " f"val size {len(val_set)}, ")

    point = train_set.__getitem__(0)
    weight_shapes = tuple(w.shape[:2] for w in point[0].weights)
    bias_shapes = tuple(b.shape[:1] for b in point[0].biases)

    logging.info(f"weight shapes: {weight_shapes}, bias shapes: {bias_shapes}")

    model = DWSModelForClassification(
        weight_shapes=weight_shapes,
        bias_shapes=bias_shapes,
        input_features=1,
        hidden_dim=args.dim_hidden,
        n_hidden=args.n_hidden,
        reduction=args.reduction,
        n_classes=args.embedding_dim,
        n_fc_layers=args.n_fc_layers,
        set_layer=args.set_layer,
        n_out_fc=args.n_out_fc,
        dropout_rate=args.do_rate,
        bn=args.add_bn,
        diagonal=args.diagonal,
    ).to(device)

    projection = nn.Sequential(
        nn.Linear(args.embedding_dim, args.embedding_dim),
        nn.ReLU(),
        nn.Linear(args.embedding_dim, args.embedding_dim),
    ).to(device)

    logging.info(f"number of parameters: {count_parameters(model)}")

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(projection.parameters()),
        lr=lr,
        amsgrad=True,
        weight_decay=5e-4,
    )

    epoch_iter = trange(epochs)

    criterion = nn.CrossEntropyLoss()

    test_acc, test_loss = -1.0, -1.0
    for epoch in epoch_iter:
        print(len(train_loader))
        for i, batch_list in enumerate(train_loader):
            batch = batch_list[0]
            aug_batch = batch_list[1]
            model.train()
            optimizer.zero_grad()

            batch = batch.to(device)
            aug_batch = aug_batch.to(device)
            inputs = (
                tuple(
                    torch.cat([w, aug_w])
                    for w, aug_w in zip(batch.weights, aug_batch.weights)
                ),
                tuple(
                    torch.cat([b, aug_b])
                    for b, aug_b in zip(batch.biases, aug_batch.biases)
                ),
            )
            features = model(inputs)
            zs = projection(features)
            logits, labels = info_nce_loss(zs, args.temperature)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            if args.wandb:
                log = {
                    "train/loss": loss.item(),
                }
                wandb.log(log)

            epoch_iter.set_description(
                f"[{epoch} {i+1}], train loss: {loss.item():.3f}"
            )

        if (epoch + 1) % eval_every == 0:
            torch.save(model.state_dict(), f"sslmodel_mnist_{epoch}.pt")


if __name__ == "__main__":
    parser = ArgumentParser("SSL trainer")

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="temperature",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=128,
        help="embedding dimension",
    )

    parser.add_argument(
        "--reduction",
        type=str,
        default="max",
        choices=["mean", "sum", "max"],
        help="reduction strategy",
    )
    parser.add_argument(
        "--dim_hidden",
        type=int,
        default=128,
        help="dim hidden layers",
    )
    parser.add_argument(
        "--n_hidden",
        type=int,
        default=4,
        help="num hidden layers",
    )
    parser.add_argument(
        "--n_fc_layers",
        type=int,
        default=1,
        help="num linear layers at each ff block",
    )
    parser.add_argument(
        "--n_out_fc",
        type=int,
        default=1,
        help="num linear layers at final layer (invariant block)",
    )
    parser.add_argument(
        "--set_layer",
        type=str,
        default="sab",
        choices=["sab", "ds"],
        help="set layer",
    )
    parser.add_argument(
        "--statistics_path",
        type=str,
        help="path to dataset statistics",
    )
    parser.add_argument(
        "--augmentation", type=str2bool, default=True, help="use augmentation"
    )
    parser.add_argument(
        "--permutation", type=str2bool, default=False, help="use permutations"
    )
    parser.add_argument(
        "--normalize", type=str2bool, default=True, help="normalize data"
    )
    parser.add_argument(
        "--diagonal", type=str2bool, default=False, help="diagonal DWSNet"
    )
    parser.add_argument("--do_rate", type=float, default=0.0, help="dropout rate")
    parser.add_argument(
        "--add_bn", type=str2bool, default=False, help="add batch norm layers"
    )
    parser.add_argument(
        "--smooth", type=str2bool, default=True, help="smooth augmentation"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="path for dataset",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=500,
        help="num epochs",
    )
    parser.add_argument("--lr", type=float, default=5e-3, help="inner learning rate")
    parser.add_argument("--batch_size", type=int, default=512, help="batch size")
    # General
    parser.add_argument(
        "--seed", type=int, default=42, help="seed value for 'set_seed'"
    )
    parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
    # Wandb
    parser.add_argument(
        "--wandb_project", type=str, default="equi_pretrain", help="wandb project name"
    )
    parser.add_argument("--wandb_entity", type=str, help="wandb entity name")
    parser.add_argument("--wandb", dest="wandb", action="store_true")
    parser.add_argument("--no-wandb", dest="wandb", action="store_false")
    parser.set_defaults(wandb=False)

    args = parser.parse_args()

    # set seed
    set_seed(args.seed)
    # wandb
    if args.wandb:
        name = (
            f"model_embedding_{args.model}_lr_{args.lr}_hid_dim_{args.dim_hidden}_reduction_{args.reduction}"
            f"_bs_{args.batch_size}_seed_{args.seed}"
        )
    wandb.init(
        name=name,
        settings=wandb.Settings(start_method="fork"),
    )
    wandb.config.update(args)

    device = get_device(gpus=args.gpu)

    main(
        path=args.data_path,
        epochs=args.n_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=device,
    )
