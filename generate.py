import argparse
import gc
import os
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from data.inr_dataset import DetailedINRDataset
from diffusers import DDIMScheduler
from models.conditional_diffusion import ConditionalDiffusionINR
from models.feature_extractor import create_equi_feature_extractor
from models.nn.inr import INR
from models.transformer import Transformer
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision.utils import save_image


def generate(test_name):
    cfg = OmegaConf.load(f"results/{test_name}/config.yaml")
    # Set seed
    set_seed(cfg.seed)
    accelerator = Accelerator()
    print(accelerator.device)
    results_folder = f"results/{test_name}"
    images_folder = f"{results_folder}/test_images"
    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(images_folder, exist_ok=True)

    # Load dataset config
    dataset_config_path = Path('./configs/dataset_config.yaml')
    dataset_config = OmegaConf.load(dataset_config_path)
    dataset_params = dataset_config[cfg.data.dataset]

    # Initialize dataset
    dataset = DetailedINRDataset(
        **dataset_params,
        normalize=True,
        split="train",
        dataset_name=cfg.data.dataset,
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0,
        mix_up=False,
        layer_swap=False,
        activation_perturbation=False,
        resolution_simulation=False,
        permutation=False,
        pattern_augment=False,
        smooth=True,
    )
    dataloader = DataLoader(
        dataset, batch_size=cfg.few_shot.training.batch_size, shuffle=False
    )

    point = dataset.__getitem__(0)
    weight_shapes = tuple(w.shape[:2] for w in point[0].weights)
    bias_shapes = tuple(b.shape[:1] for b in point[0].biases)

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
    feature_extractor = create_equi_feature_extractor(
        weight_shapes,
        bias_shapes,
        cfg.model.input_features,
        cfg.model.hidden_dim,
        n_hidden=4,
        n_classes=cfg.model.n_classes,
    )
    transformer = Transformer(
        layers,
        layer_names,
        **cfg.model.transformer_config,
    )

    model = ConditionalDiffusionINR(
        transformer,
        DDIMScheduler(**cfg.model.scheduler_config),
        feature_extractor,
        stats=dataset.stats,
    )

    # TODO: Load model from checkpoint, modify for different generation
    model.load_state_dict(torch.load(f"{results_folder}/checkpoints/model.pth"))
    model, dataloader = accelerator.prepare(model, dataloader)

    model = model.module if hasattr(model, 'module') else model
    device = next(model.parameters()).device

    # Generate samples
    for i, batch_list in enumerate(dataloader):
        for k, batch in enumerate(batch_list):
            weights = tuple(w.to(device) for w in batch.weights)
            biases = tuple(b.to(device) for b in batch.biases)
            with torch.no_grad():
                generated_weights, model_output = model.generate(
                    weights=weights,
                    biases=biases,
                )

            # Generate images from INRs
            samples = dataset.generate_image_from_weights(generated_weights)

            # Save generated images
            for j, sample in enumerate(samples):
                save_image(
                    sample,
                    f"{results_folder}/test_images/sample_{i}_{k}_{j}_{batch.label[j]}.png",
                )

        # Clear CUDA cache
        torch.cuda.empty_cache()
        gc.collect()

    # Free up memory
    del model, unetmodel, feature_extractor, dataset, dataloader, point
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_name",
        type=str,
    )
    args = parser.parse_args()
    generate(args.test_name)
