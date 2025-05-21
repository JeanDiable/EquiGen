import torch
from torch import nn

from models.nn.models import DWSModelForClassification


class EquiFeatureExtractor(nn.Module):
    def __init__(
        self,
        weight_shapes,
        bias_shapes,
        input_features,
        hidden_dim,
        n_classes=9,
        n_hidden=2,
    ):
        super().__init__()
        self.equi = DWSModelForClassification(
            weight_shapes=weight_shapes,
            bias_shapes=bias_shapes,
            input_features=input_features,
            hidden_dim=hidden_dim,
            n_hidden=n_hidden,
            n_classes=n_classes,  # We use hidden_dim as the output dimension for features
        )

    def forward(self, x):
        out, features = self.equi(x, return_equiv=True)
        return features


def create_equi_feature_extractor(
    weight_shapes, bias_shapes, input_features, hidden_dim, n_classes=9, n_hidden=4
):
    return EquiFeatureExtractor(
        weight_shapes, bias_shapes, input_features, hidden_dim, n_classes, n_hidden
    )


def load_pretrained_weights(self, path_to_checkpoint):
    '''
    Load pretrained weights for the DWSNet feature extractor
    '''
    checkpoint = torch.load(path_to_checkpoint)
    self.equi.load_state_dict(checkpoint)
