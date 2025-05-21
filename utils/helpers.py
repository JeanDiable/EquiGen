import logging
import os
from typing import List, Tuple, Union

import numpy as np
import torch
from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import save_image


def weights_to_inr_transformer(flattened_weights, original_shapes, batch_size):
    '''
    Convert flattened weights and biases to INR weights and biases
    '''
    inrs = []
    start_idx = 0
    for i in range(batch_size):
        weights = []
        biases = []

        for j in range(len(original_shapes) // 2):
            shape1 = original_shapes[j]
            shape2 = original_shapes[j + len(original_shapes) // 2]
            num_elements1 = np.prod(shape1)
            num_elements2 = np.prod(shape2)

            weight = flattened_weights[start_idx : start_idx + num_elements1].reshape(
                shape1
            )
            weights.append(weight)  # This is a bias vector
            bias = flattened_weights[
                start_idx + num_elements1 : start_idx + num_elements1 + num_elements2
            ].reshape(shape2)
            biases.append(bias)
            start_idx += num_elements1 + num_elements2
        inrs.append((weights, biases))
    # Reshape inrs into a tuple of (weights, biases)
    reshaped_inrs = ([], [])
    for i in range(len(inrs[0][0])):  # 3
        # Collect weights for this layer from all batches
        weights_layer = torch.stack([inr[0][i] for inr in inrs])
        reshaped_inrs[0].append(weights_layer)

        # Collect biases for this layer from all batches
        biases_layer = torch.stack([inr[1][i] for inr in inrs])
        reshaped_inrs[1].append(biases_layer)

    # Convert lists to tuples
    reshaped_inrs = (tuple(reshaped_inrs[0]), tuple(reshaped_inrs[1]))
    inrs = reshaped_inrs
    return inrs


def prepare_diffusion_input_transformer(weights, biases, batch_size):
    '''
    Prepare weights and biases for UNet1DModel and Transformer
    '''
    device = weights[0].device
    input = []
    # Flatten and concatenate weights and biases pairly, layer by layer
    for i in range(weights[0].shape[0]):  # batch size
        w = [weights[j][i].flatten() for j in range(len(weights))]
        b = [biases[j][i].flatten() for j in range(len(biases))]
        for w, b in zip(w, b):
            input.append(torch.cat([w, b]))
    input = torch.hstack(input)
    input = input.reshape(batch_size, -1)

    return input


def make_coordinates(
    shape: Union[Tuple[int], List[int]],
    bs: int,
    coord_range: Union[Tuple[int], List[int]] = (-1, 1),
) -> torch.Tensor:
    '''
    Make coordinates for INR
    '''
    x_coordinates = np.linspace(coord_range[0], coord_range[1], shape[0])
    y_coordinates = np.linspace(coord_range[0], coord_range[1], shape[1])
    x_coordinates, y_coordinates = np.meshgrid(x_coordinates, y_coordinates)
    x_coordinates = x_coordinates.flatten()
    y_coordinates = y_coordinates.flatten()
    coordinates = np.stack([x_coordinates, y_coordinates]).T
    coordinates = np.repeat(coordinates[np.newaxis, ...], bs, axis=0)
    return torch.from_numpy(coordinates).type(torch.float)


def set_up_dataset(dataset_name):
    # Download and save images
    if dataset_name == "mnist":
        mnist_dataset = MNIST(
            root='./dataset', train=True, download=True, transform=ToTensor()
        )
        os.makedirs("dataset/mnist_images", exist_ok=True)
        for i, (img, _) in enumerate(mnist_dataset):
            save_image(img, f"dataset/mnist_images/mnist_{i}.png")
    elif dataset_name == "cifar10":
        cifar10_dataset = CIFAR10(
            root='./dataset', train=True, download=True, transform=ToTensor()
        )
        os.makedirs("dataset/cifar10_images", exist_ok=True)
        for i, (img, _) in enumerate(cifar10_dataset):
            save_image(img, f"dataset/cifar10_images/cifar10_{i}.png")
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")


def get_device(no_cuda=False, gpus="0"):
    return torch.device(
        f"cuda:{gpus}" if torch.cuda.is_available() and not no_cuda else "cpu"
    )


def set_logger():
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
