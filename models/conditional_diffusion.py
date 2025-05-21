import copy

import torch
from torch import nn
from torch.nn import functional as F

from utils.ema import EMA
from utils.helpers import (
    count_parameters,
    prepare_diffusion_input_transformer,
    weights_to_inr,
    weights_to_inr_transformer,
)


class ConditionalDiffusionINR(nn.Module):
    def __init__(
        self,
        model,
        scheduler,
        feature_extractor,
        stats=None,
    ):
        super().__init__()
        self.model = model
        self.scheduler = scheduler
        self.feature_extractor = feature_extractor
        self.stats = stats
        self.ema = EMA(beta=0.999)
        self.ema_model = copy.deepcopy(self.model)

    def forward(self, x, timesteps, context):

        features = self.feature_extractor(context)
        # fit the shape to the transformer model
        features = features.repeat(1, 8)
        features = F.pad(features, (0, 1024 - features.shape[1]))
        features = features.unsqueeze(1)

        sample = self.model(x, timesteps, x_prev=x, condition=features)

        return sample

    @torch.no_grad()
    def generate(
        self,
        weights,
        biases,
        num_inference_steps=100,
        use_condition=True,
    ):
        '''
        Generate new INR weights from the input weights and biases
        '''
        batch_size = weights[0].shape[0]
        device = weights[0].device

        features = self.feature_extractor((weights, biases))
        # fit the shape to the transformer model
        features = features.repeat(1, 8)
        features = F.pad(features, (0, 1024 - features.shape[1]))
        features = features.unsqueeze(1)

        # trigger this if it is few-shot learning
        # TODO adding configuration for subspace disturbance
        features = features + torch.randn_like(features) * 0.3

        if not use_condition:
            features = torch.zeros_like(features)

        sample_shape = prepare_diffusion_input_transformer(
            weights, biases, weights[0].shape[0]
        ).shape

        sample = torch.randn(sample_shape, device=device)

        # Set number of inference steps
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.scheduler.timesteps:
            # Duplicate sample for classifier-free guidance
            sample_input = sample
            t_input = torch.cat([t.unsqueeze(0)] * batch_size).flatten()

            model_output = self.ema_model(
                sample_input,
                t_input.unsqueeze(1),
                x_prev=sample_input,
                condition=features,
            )

            sample = sample.to(device)
            model_output = model_output.to(device)
            alpha_t = self.scheduler.alphas_cumprod[t].to(device)
            noise_pred = (sample - alpha_t.sqrt().unsqueeze(-1) * model_output) / (
                1 - alpha_t
            ).sqrt().unsqueeze(-1)

            sample = self.scheduler.step(noise_pred, t, sample).prev_sample

        shapes = [w.shape[1:] for w in weights] + [b.shape[1:] for b in biases]
        sample = sample.flatten()

        inr_weights = weights_to_inr_transformer(sample, shapes, weights[0].shape[0])
        # Denormalize the generated weights and biases
        inr_weights = self.denormalize(inr_weights)

        return inr_weights, model_output

    def count_parameters(self):
        return count_parameters(self.model)

    def denormalize(self, weights_and_biases):
        # Denormalize weights and biases
        weights, biases = weights_and_biases

        # Ensure all tensors are on the same device
        device = weights[0].device
        wm = [m.to(device) for m in self.stats["weights"]["mean"]]
        ws = [s.to(device) for s in self.stats["weights"]["std"]]
        bm = [m.to(device) for m in self.stats["biases"]["mean"]]
        bs = [s.to(device) for s in self.stats["biases"]["std"]]

        # Denormalize weights
        denorm_weights = tuple(
            w * s.unsqueeze(0) + m.unsqueeze(0) for w, m, s in zip(weights, wm, ws)
        )

        # Denormalize biases
        denorm_biases = tuple(
            b * s.unsqueeze(0) + m.unsqueeze(0) for b, m, s in zip(biases, bm, bs)
        )

        return (denorm_weights, denorm_biases)
