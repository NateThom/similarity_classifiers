import pandas as pd
import torch
from pytorch_lightning import LightningModule
import numpy as np

# SimCLR
from simclr import SimCLR
from simclr_modules import NT_Xent, get_resnet

class ContrastiveLearning(LightningModule):
    def __init__(self, args):
        super().__init__()

        # self.hparams = args
        self.args = args

        # initialize ResNet
        self.encoder = get_resnet(self.args.resnet, pretrained=self.args.pretrain)
        self.n_features = self.encoder.fc.in_features  # get dimensions of fc layer
        self.model = SimCLR(self.encoder, self.args.h_dim, self.args.projection_dim, self.n_features,
                            self.args.n_classes)
        self.test_outputs = np.array([])
        self.criterion = NT_Xent(self.args.batch_size, self.args.temperature, world_size=1)

    def forward(self, x_i, x_j):
        h_i, h_j, z_i, z_j = self.model(x_i, x_j)
        return h_i, h_j, z_i, z_j

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        if self.args.dataset == "VGG_Face":
            (x_i, x_j) = batch
        else:
            (x_i, x_j), _ = batch
        h_i, h_j, z_i, z_j = self.forward(x_i, x_j)
        loss = self.criterion(z_i, z_j)
        self.log("Training Loss", loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return loss

    def test_step(self, test_batch, batch_idx):
        (x_i, _), _ = test_batch
        h_i, h_j, z_i, z_j = self.forward(x_i, x_i)
        h_i = h_i.cpu().numpy()

        if len(self.test_outputs) == 0:
            self.test_outputs = h_i
        else:
            self.test_outputs = np.append(self.test_outputs, h_i, axis=0)

    def test_epoch_end(self, outputs):
        output_csv = pd.DataFrame(self.test_outputs)
        output_csv.to_csv(self.args.csv_path + self.args.model_file[:-4] + "csv", header=False, index=False)

    # def configure_criterion(self):
    #     criterion = NT_Xent(self.args.batch_size, self.args.temperature)
    #     return criterion

    def configure_optimizers(self):
        scheduler = None
        if self.args.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
                                         amsgrad=self.args.amsgrad)
        elif self.args.optimizer == "LARS":
            # optimized using LARS with linear learning rate scaling
            # (i.e. LearningRate = 0.3 × BatchSize/256) and weight decay of 10−6.
            learning_rate = 0.3 * args.batch_size / 256
            optimizer = LARS(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=args.weight_decay,
                exclude_from_weight_decay=["batch_normalization", "bias"],
            )

            # "decay the learning rate with the cosine decay schedule without restarts"
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, args.epochs, eta_min=0, last_epoch=-1
            )
        else:
            raise NotImplementedError

        if scheduler:
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return {"optimizer": optimizer}