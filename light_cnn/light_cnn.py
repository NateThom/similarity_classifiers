import torch.nn as nn
import torch.nn.functional as F
import torch

from modules.resnet_hacks import modify_resnet_model
from modules.identity import Identity


class LightCnn(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, encoder, n_features, n_classes):
        super(LightCnn, self).__init__()

        self.encoder = encoder
        self.n_features = n_features
        self.n_classes = n_classes
        # self.n_classes = 128
        # self.n_classes = 226

        # Replace the fc layer with an Identity function
        self.encoder.fc = nn.Linear(self.n_features, self.n_classes)
        # self.encoder.pos_fc = Identity()

        # # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        # self.projector = nn.Sequential(
        #     nn.Linear(self.h_dim, self.n_classes, bias=False),
        #     nn.ReLU(),
        #     nn.Linear(self.n_classes, projection_dim, bias=False),
        # )

    def forward(self, x_i):
        h_i = self.encoder(x_i)
        h_i = F.softmax(h_i, 1)

        return h_i
