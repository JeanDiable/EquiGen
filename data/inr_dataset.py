import copy
import json
import random
from typing import NamedTuple, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from models.nn.inr import INR
from utils.helpers import make_coordinates
from utils.smooth_augment import PermutationManager


class Batch(NamedTuple):
    weights: Tuple
    biases: Tuple
    label: Union[torch.Tensor, int]

    def _assert_same_len(self):
        assert len(set([len(t) for t in self])) == 1

    def as_dict(self):
        return self._asdict()

    def to(self, device):
        """move batch to device"""
        return self.__class__(
            weights=tuple(w.to(device) for w in self.weights),
            biases=tuple(w.to(device) for w in self.biases),
            label=self.label.to(device),
        )

    def __len__(self):
        return len(self.weights[0])


class INRDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        path,
        split="train",
        normalize=False,
        augmentation=False,
        permutation=False,
        statistics_path="dataset/statistics.pth",
        translation_scale=0.2,
        rotation_degree=30,
        noise_scale=0,
        drop_rate=0,
        resize_scale=0.0,
        pos_scale=0.0,
        quantile_dropout=0.0,
        class_mapping=None,
        color_jitter=False,
        brightness=0.1,
        contrast=0.1,
        saturation=0.1,
        activation_perturbation=False,
        smooth=False,
        inr=INR,
        inr_kwargs={"n_layers": 3, "in_dim": 2, "up_scale": 32, "out_channels": 3},
    ):
        # assert split in ["test", "train"]
        self.split = split
        self.dataset = json.load(open(path, "r"))[self.split]

        self.augmentation = augmentation
        self.permutation = permutation
        self.normalize = normalize
        if self.normalize:
            self.stats = torch.load(statistics_path, map_location="cpu")

        self.translation_scale = translation_scale
        self.rotation_degree = rotation_degree
        self.noise_scale = noise_scale
        self.drop_rate = drop_rate
        self.resize_scale = resize_scale
        self.pos_scale = pos_scale
        self.quantile_dropout = quantile_dropout
        self.color_jitter = color_jitter
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.activation_perturbation = activation_perturbation
        if class_mapping is not None:
            self.class_mapping = class_mapping
            self.dataset["label"] = [
                self.class_mapping[l] for l in self.dataset["label"]
            ]
        self.smooth = smooth
        self.inr_kwargs = inr_kwargs
        self.inr = inr

    def __len__(self):
        return len(self.dataset["label"])

    def _normalize(self, weights, biases):
        wm, ws = self.stats["weights"]["mean"], self.stats["weights"]["std"]
        bm, bs = self.stats["biases"]["mean"], self.stats["biases"]["std"]

        weights = tuple((w - m) / s for w, m, s in zip(weights, wm, ws))
        biases = tuple((w - m) / s for w, m, s in zip(biases, bm, bs))

        return weights, biases

    @staticmethod
    def rotation_mat(degree=30.0):
        angle = torch.empty(1).uniform_(-degree, degree)
        angle_rad = angle * (torch.pi / 180)
        rotation_matrix = torch.tensor(
            [
                [torch.cos(angle_rad), -torch.sin(angle_rad)],
                [torch.sin(angle_rad), torch.cos(angle_rad)],
            ]
        )
        return rotation_matrix

    def _augment(self, weights, biases):
        """Augmentations for MLP (and some INR specific ones)

        :param weights:
        :param biases:
        :return:
        """
        new_weights, new_biases = list(weights), list(biases)
        # translation
        translation = torch.empty(weights[0].shape[0]).uniform_(
            -self.translation_scale, self.translation_scale
        )
        order = random.sample(range(1, len(weights)), 1)[0]
        bias_res = translation
        i = 0
        for i in range(order):
            bias_res = bias_res @ weights[i]

        new_biases[i] += bias_res

        # rotation
        if new_weights[0].shape[0] == 2:
            rot_mat = self.rotation_mat(self.rotation_degree)
            new_weights[0] = rot_mat @ new_weights[0]

        # noise
        new_weights = [w + w.std() * self.noise_scale for w in new_weights]
        new_biases = [
            b + b.std() * self.noise_scale if b.shape[0] > 1 else b for b in new_biases
        ]

        # dropout
        new_weights = [F.dropout(w, p=self.drop_rate) for w in new_weights]
        new_biases = [F.dropout(w, p=self.drop_rate) for w in new_biases]

        # scale
        # todo: can also apply to deeper layers
        rand_scale = 1 + (torch.rand(1).item() - 0.5) * 2 * self.resize_scale
        new_weights[0] = new_weights[0] * rand_scale

        # positive scale
        if self.pos_scale > 0:
            for i in range(len(new_weights) - 1):
                # todo: we do a lot of duplicated stuff here
                out_dim = new_biases[i].shape[0]
                scale = torch.from_numpy(
                    np.random.uniform(
                        1 - self.pos_scale, 1 + self.pos_scale, out_dim
                    ).astype(np.float32)
                )
                inv_scale = 1.0 / scale
                new_weights[i] = new_weights[i] * scale
                new_biases[i] = new_biases[i] * scale
                new_weights[i + 1] = (new_weights[i + 1].T * inv_scale).T
        if self.quantile_dropout > 0:
            do_q = torch.empty(1).uniform_(0, self.quantile_dropout)
            q = torch.quantile(
                torch.cat([v.flatten().abs() for v in new_weights + new_biases]), q=do_q
            )
            new_weights = [torch.where(w.abs() < q, 0, w) for w in new_weights]
            new_biases = [torch.where(w.abs() < q, 0, w) for w in new_biases]

        return tuple(new_weights), tuple(new_biases)

    def _activation_perturbation(self, weights, biases, perturb_range=0.01):
        new_biases = list(biases)
        for i in range(len(new_biases) - 1):  # Don't perturb the output layer
            perturb = torch.empty_like(new_biases[i]).uniform_(
                -perturb_range, perturb_range
            )
            new_biases[i] = new_biases[i] + perturb
        return weights, tuple(new_biases)

    def _color_augment(self, weights, biases):
        # Assuming the last layer represents RGB channels
        last_weight = weights[-1]
        last_bias = biases[-1]

        # Apply brightness adjustment
        if self.brightness > 0:
            brightness_factor = torch.empty(1).uniform_(
                1 - self.brightness, 1 + self.brightness
            )
            last_weight = last_weight * brightness_factor
            last_bias = last_bias * brightness_factor

        # Apply contrast adjustment
        if self.contrast > 0:
            contrast_factor = torch.empty(1).uniform_(
                1 - self.contrast, 1 + self.contrast
            )
            last_weight = (
                last_weight - last_weight.mean()
            ) * contrast_factor + last_weight.mean()
            last_bias = (
                last_bias - last_bias.mean()
            ) * contrast_factor + last_bias.mean()

        # Apply saturation adjustment
        if self.saturation > 0:
            saturation_factor = torch.empty(1).uniform_(
                1 - self.saturation, 1 + self.saturation
            )
            grayscale = (
                0.2989 * last_weight[0]
                + 0.5870 * last_weight[1]
                + 0.1140 * last_weight[2]
            )
            last_weight = torch.lerp(
                grayscale.unsqueeze(0).expand_as(last_weight),
                last_weight,
                saturation_factor,
            )

        new_weights = list(weights)
        new_biases = list(biases)
        new_weights[-1] = last_weight
        new_biases[-1] = last_bias

        return tuple(new_weights), tuple(new_biases)

    @staticmethod
    def _permute(weights, biases):
        new_weights = [None] * len(weights)
        new_biases = [None] * len(biases)
        assert len(weights) == len(biases)

        perms = []
        for i, w in enumerate(weights):
            if i != len(weights) - 1:
                perms.append(torch.randperm(w.shape[1]))

        for i, (w, b) in enumerate(zip(weights, biases)):
            if i == 0:
                new_weights[i] = w[:, perms[i], :]
                new_biases[i] = b[perms[i], :]
            elif i == len(weights) - 1:
                new_weights[i] = w[perms[-1], :, :]
                new_biases[i] = b
            else:
                new_weights[i] = w[perms[i - 1], :, :][:, perms[i], :]
                new_biases[i] = b[perms[i], :]
        return new_weights, new_biases

    def __getitem__(self, item):
        path = self.dataset["path"][item]
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)

        if isinstance(state_dict, dict) and 'params' in state_dict:
            state_dict = state_dict['params']

        if self.smooth:
            example_tensor = torch.randn(1, 2)
            inr = self.inr(**self.inr_kwargs)
            inr.load_state_dict(state_dict)

            perm_manager = PermutationManager(inr, example_tensor)
            smoothed_inr = perm_manager()
            smoothed_state_dict = smoothed_inr.state_dict()
        else:
            smoothed_state_dict = state_dict

        weights = tuple(
            [v.permute(1, 0) for w, v in smoothed_state_dict.items() if "weight" in w]
        )
        biases = tuple([v for w, v in smoothed_state_dict.items() if "bias" in w])
        label = int(self.dataset["label"][item])
        # Create a list to store all versions (original + augmented)

        # TODO: Debug copy problem
        all_versions = [(weights, biases, label)]
        weights_new = copy.deepcopy(weights)
        biases_new = copy.deepcopy(biases)
        weights_new2 = copy.deepcopy(weights)
        biases_new2 = copy.deepcopy(biases)
        weights_new3 = copy.deepcopy(weights)
        biases_new3 = copy.deepcopy(biases)
        weights_new4 = copy.deepcopy(weights)
        biases_new4 = copy.deepcopy(biases)

        if self.augmentation:
            augmented_weights, augmented_biases = self._augment(weights_new, biases_new)
            all_versions.append((augmented_weights, augmented_biases, label))
        if self.color_jitter:
            color_augmented_weights, color_augmented_biases = self._color_augment(
                weights_new2, biases_new2
            )
            all_versions.append(
                (color_augmented_weights, color_augmented_biases, label)
            )
        if self.activation_perturbation:
            perturbed_weights, perturbed_biases = self._activation_perturbation(
                weights_new3, biases_new3
            )
            all_versions.append((perturbed_weights, perturbed_biases, label))
        if self.permutation:
            permuted_weights, permuted_biases = self._permute(weights_new4, biases_new4)
            all_versions.append((permuted_weights, permuted_biases, label))

        processed_versions = []
        for version_weights, version_biases, version_label in all_versions:
            # add feature dim
            version_weights = tuple([w.unsqueeze(-1) for w in version_weights])
            version_biases = tuple([b.unsqueeze(-1) for b in version_biases])

            if self.normalize:
                version_weights, version_biases = self._normalize(
                    version_weights, version_biases
                )
            processed_versions.append(
                Batch(
                    weights=version_weights, biases=version_biases, label=version_label
                )
            )

        return processed_versions


class DetailedINRDataset(INRDataset):
    def __init__(
        self,
        path,
        split="train",
        normalize=False,
        augmentation=False,
        permutation=False,
        statistics_path="dataset/statistics.pth",
        translation_scale=0.4,
        rotation_degree=45,
        noise_scale=5e-2,
        drop_rate=5e-3,
        resize_scale=0.2,
        pos_scale=0.0,
        quantile_dropout=0.0,
        class_mapping=None,
        image_size=(28, 28),
        inr_class=INR,
        inr_kwargs={"n_layers": 3, "in_dim": 2, "up_scale": 16},
        dataset_name="mnist",
        color_jitter=False,
        brightness=0.1,
        contrast=0.1,
        saturation=0.1,
        activation_perturbation=False,
        smooth=True,
    ):
        super().__init__(
            path,
            split,
            normalize,
            augmentation,
            permutation,
            statistics_path,
            translation_scale,
            rotation_degree,
            noise_scale,
            drop_rate,
            resize_scale,
            pos_scale,
            quantile_dropout,
            class_mapping,
            color_jitter,
            brightness,
            contrast,
            saturation,
            activation_perturbation,
            smooth,
        )
        self.image_size = image_size
        self.inr_class = inr_class
        self.inr_kwargs = inr_kwargs
        self.dataset_name = dataset_name
        self.dataset_path = f"dataset/{dataset_name}_images"
        self.color_jitter = color_jitter
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def generate_image_from_weights(self, weights_and_biases):
        '''
        Generate an image from the INR weights and biases
        '''
        images = []
        for i in range(weights_and_biases[0][0].shape[0]):
            weights = tuple(
                weights_and_biases[0][j][i] for j in range(len(weights_and_biases[0]))
            )
            biases = tuple(
                weights_and_biases[1][j][i] for j in range(len(weights_and_biases[1]))
            )
            inr = self.inr_class(**self.inr_kwargs)
            state_dict = {}
            for i, (w, b) in enumerate(zip(weights, biases)):
                state_dict[f'seq.{i}.weight'] = w.squeeze(-1).permute(1, 0)
                state_dict[f'seq.{i}.bias'] = b.squeeze(-1)

            inr.load_state_dict(state_dict)
            inr.eval()
            input_coords = make_coordinates(self.image_size, 1)
            with torch.no_grad():
                image = inr(input_coords)
                image = image.view(*self.image_size, -1)
                image = image.permute(2, 0, 1)
                if image.shape[0] == 3:
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    image = image * std + mean
                    image = torch.clamp(image, 0, 1)
            images.append(image)
        return images
