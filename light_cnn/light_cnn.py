import torch.nn as nn
import torch.nn.functional as F

class LightCnn(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, encoder, n_features, n_classes):
        super(LightCnn, self).__init__()

        self.encoder = encoder
        self.n_features = n_features
        self.n_classes = n_classes
        # self.n_classes = 3
        # self.n_classes = 5
        # self.n_classes = 128
        # self.n_classes = 226

        # Replace the original 1000 class output fc layer with n_classes output fc layer
        self.encoder.fc = nn.Linear(self.n_features, self.n_classes)

    def forward(self, x_i):
        h_i = self.encoder(x_i)
        h_i = F.softmax(h_i, 1)

        return h_i
