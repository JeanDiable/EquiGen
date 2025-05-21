import os
import sys

import hydra
import numpy as np
import torch

sys.path.append(os.path.abspath(__file__))
from pathlib import Path

from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import DDIMScheduler
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from data.inr_dataset import DetailedINRDataset
from models.conditional_diffusion import ConditionalDiffusionINR
from models.feature_extractor import create_equi_feature_extractor
from models.nn.inr import INR
from models.transformer import Transformer
from utils.helpers import (
    count_parameters,
    prepare_diffusion_input_transformer,
    weights_to_inr_transformer,
)


@hydra.main(config_path="configs", config_name="default", version_base="1.3")
def train(cfg: DictConfig):
    dataset_config_path = Path('./configs/dataset_config.yaml')
    dataset_config = OmegaConf.load(dataset_config_path)
    dataset_name = cfg.data.dataset
    dataset_params = dataset_config[dataset_name]

    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        log_with="wandb" if cfg.wandb.enabled else None,
    )

    if cfg.wandb.enabled:
        accelerator.init_trackers(
            project_name=cfg.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": {"entity": cfg.wandb.entity}, "name": cfg.wandb.name},
        )
        results_folder = f"results/{cfg.wandb.name}"
    else:
        results_folder = f"results/local_run_{cfg.seed}"
    os.makedirs(f"{results_folder}/results", exist_ok=True)
    os.makedirs(f"{results_folder}/checkpoints", exist_ok=True)
    if cfg.is_few_shot:
        os.makedirs(f"{results_folder}/results/few_shot", exist_ok=True)
    # store the config file in the results folder
    with open(f"{results_folder}/config.yaml", "w") as f:
        OmegaConf.save(cfg, f)
    # set seed
    set_seed(cfg.seed)

    # Initialize dataset and dataloader
    accelerator.print('Initializing dataset and dataloader')
    train_dataset = DetailedINRDataset(
        **dataset_params,
        normalize=True,
        augmentation=False,
        split='train',
        dataset_name=dataset_name,
        smooth=True,
    )
    eval_dataset = DetailedINRDataset(
        **dataset_params,
        normalize=True,
        split='test',
        smooth=True,
        dataset_name=dataset_name,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=cfg.training.num_workers,
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=cfg.training.batch_size, shuffle=False
    )
    if cfg.is_few_shot:
        dataset_params = dataset_config[cfg.few_shot.dataset]
        few_shot_dataset = DetailedINRDataset(
            **dataset_params,
            normalize=True,
            split='train',
            smooth=True,
            dataset_name=cfg.few_shot.dataset,
            color_jitter=False,
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0,
            activation_perturbation=False,
            augmentation=False,
        )
        eval_dataset = DetailedINRDataset(
            **dataset_params,
            normalize=True,
            split='test',
            smooth=True,
            dataset_name=cfg.few_shot.dataset,
            color_jitter=False,
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0,
            activation_perturbation=False,
            augmentation=False,
        )
        train_loader = DataLoader(
            few_shot_dataset, batch_size=cfg.few_shot.training.batch_size, shuffle=True
        )
        eval_loader = DataLoader(
            eval_dataset, batch_size=cfg.few_shot.training.batch_size, shuffle=False
        )

    accelerator.print('Initializing equivariant encoder')
    # Initialize equivariant encoder
    point = train_dataset.__getitem__(0)
    weight_shapes = tuple(w.shape[:2] for w in point[0].weights)
    bias_shapes = tuple(b.shape[:1] for b in point[0].biases)

    feature_extractor = create_equi_feature_extractor(
        weight_shapes,
        bias_shapes,
        cfg.model.input_features,
        cfg.model.hidden_dim,
        n_hidden=4,
        n_classes=cfg.model.n_classes,
    )

    accelerator.print('Initializing equivariant conditioned diffusion model')
    # Initialize equivariant conditioned diffusion model

    inr = INR(
        **dataset_params['inr_kwargs'],
        pe_features=None,
        fix_pe=True,
    )
    state_dict = inr.state_dict()
    layers = []
    layer_names = []
    for l in state_dict:
        shape = state_dict[l].shape
        layers.append(np.prod(shape))
        layer_names.append(l)
    transformer = Transformer(
        layers,
        layer_names,
        **cfg.model.transformer_config,
    ).cuda()

    model = ConditionalDiffusionINR(
        transformer,
        DDIMScheduler(**cfg.model.scheduler_config),
        feature_extractor,
        stats=train_dataset.stats,
    )

    if cfg.training.checkpoint_path is not None:
        model.load_state_dict(
            {k: v for k, v in torch.load(cfg.training.checkpoint_path).items()}
        )
    if cfg.model.feature_extractor_path is not None:
        feature_extractor.load_state_dict(
            {
                f'dwsnet.{k}': v
                for k, v in torch.load(cfg.model.feature_extractor_path).items()
            }
        )
        print("Feature extractor loaded")

    feature_extractor.requires_grad = (
        cfg.model.train_feature_extractor or cfg.is_few_shot
    )
    accelerator.log({"number of parameters": count_parameters(model)})

    accelerator.print('Initializing optimizer')
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    accelerator.print('Initializing learning rate scheduler')
    # Learning rate scheduler
    if cfg.training.lr_scheduler == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.training.num_epochs, eta_min=1e-6
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.training.scheduler_step, gamma=0.9
        )

    accelerator.print('Preparing model, optimizer, and dataloader with accelerator')
    # Prepare model, optimizer, and dataloader with accelerator
    model, optimizer, train_loader, eval_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, lr_scheduler
    )

    model = model.module if hasattr(model, 'module') else model

    accelerator.print('Initializing loss function')
    # Loss function
    if cfg.training.loss_function == 'mse':
        criterion = torch.nn.MSELoss()
        criterion_few_shot = torch.nn.MSELoss()
    elif cfg.training.loss_function == 'cross_entropy':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Loss function {cfg.training.loss_function} not supported")

    accelerator.print('Starting training loop')
    # Training loop
    global_step = 0
    for epoch in tqdm(range(cfg.training.num_epochs), desc="Training Epochs"):
        model.train()
        total_loss = 0
        for step, batch_list in tqdm(enumerate(train_loader), desc="Training Steps"):
            for batch in batch_list:
                with accelerator.accumulate(model):
                    inr_weights, inr_biases = batch.weights, batch.biases

                    diffusion_input = prepare_diffusion_input_transformer(
                        inr_weights, inr_biases, batch.label.shape[0]
                    )

                    noise = torch.randn_like(diffusion_input)
                    timesteps = torch.randint(
                        0,
                        model.scheduler.num_train_timesteps,
                        (diffusion_input.shape[0],),
                        device=diffusion_input.device,
                    ).long()

                    noisy_input = model.scheduler.add_noise(
                        diffusion_input, noise, timesteps
                    )
                    timesteps = timesteps.unsqueeze(1)

                    noise_pred = model(
                        noisy_input,
                        timesteps,
                        (inr_weights, inr_biases),
                    )

                    loss = criterion(noise_pred, noise)

                    if cfg.is_few_shot:
                        # add regularization term between feature extracted from original and few shot data
                        shapes = [w.shape[1:] for w in inr_weights] + [
                            b.shape[1:] for b in inr_biases
                        ]
                        noise_pred = noise_pred.flatten()
                        inrs = weights_to_inr_transformer(
                            noise_pred, shapes, cfg.few_shot.training.batch_size
                        )
                        few_shot_weights, few_shot_biases = inrs

                        few_shot_loss = criterion_few_shot(
                            model.feature_extractor((inr_weights, inr_biases)),
                            model.feature_extractor(
                                (few_shot_weights, few_shot_biases)
                            ),
                        )
                        loss = loss + cfg.few_shot.training.lamb * few_shot_loss

                    accelerator.print(f"Loss: {loss.item()}")

                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            model.parameters(), cfg.training.max_grad_norm
                        )
                    accelerator.wait_for_everyone()
                    optimizer.step()
                    optimizer.zero_grad()
                    model.ema.step_ema(model.ema_model, model.model)

                total_loss += loss.detach().item()
                global_step += 1

                if cfg.is_few_shot:
                    eval_every = cfg.few_shot.training.eval_every
                else:
                    eval_every = cfg.training.eval_every

                if global_step % eval_every == 0:
                    evaluate(
                        model,
                        eval_dataset,
                        eval_loader,
                        accelerator,
                        global_step,
                        results_folder,
                    )

        avg_loss = total_loss / len(train_loader)
        lr_scheduler.step()

        accelerator.log(
            {
                "train_loss": avg_loss,
                "epoch": epoch,
                "learning_rate": lr_scheduler.get_last_lr()[0],
            },
            step=global_step,
        )

        if (epoch + 1) % cfg.training.save_every == 0:
            torch.save(
                model.state_dict(),
                f"{results_folder}/checkpoints/model-epoch-{epoch + 1}.pth",
            )
    accelerator.end_training()


@torch.no_grad()
def evaluate(
    model,
    dataset,
    dataloader,
    accelerator,
    step,
    results_folder,
):
    # make evaluation only on the first gpu
    if accelerator.is_main_process:
        model.eval()
        model = model.module if hasattr(model, 'module') else model
        device = next(model.parameters()).device

        # Save generated images for FID calculation
        os.makedirs(f"{results_folder}/results/generated_images", exist_ok=True)
        labels_file_path = f"{results_folder}/results/generated_images/labels.txt"
        for i, batch_list in tqdm(enumerate(dataloader)):
            for k in range(len(batch_list)):
                batch = batch_list[k]
                inr_weights, inr_biases, labels = (
                    batch.weights,
                    batch.biases,
                    batch.label,
                )
                labels = [str(label) for label in labels]

                # Save labels at each step into a file under results/generated_images
                with open(labels_file_path, 'a') as labels_file:
                    labels_file.write(f"{step}\n")
                    for label in labels:
                        labels_file.write(f"{label} ")
                    labels_file.write("\n")

                # Move tensors to the correct device
                inr_weights = tuple(w.to(device) for w in inr_weights)
                inr_biases = tuple(b.to(device) for b in inr_biases)
                generated_weights, model_output = model.generate(
                    inr_weights,
                    inr_biases,
                )
                samples = dataset.generate_image_from_weights(generated_weights)
                for j, sample in enumerate(samples):
                    save_image(
                        sample,
                        f"{results_folder}/results/generated_images/sample_{i}_{k}_{j}_{step}_{labels[j]}.png",
                    )

    model.train()


if __name__ == "__main__":
    train()
