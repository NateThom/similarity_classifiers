import argparse
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
import numpy as np
import matplotlib.pyplot as plt
import wandb
from utils import yaml_config_hook
import torchmetrics

#my code
import vgg_face_dataset
from light_cnn import LightCnn
from contrastive_learning import ContrastiveLearning

#simclr
from transformations import TransformsSimCLR
from modules import get_resnet

class SimilarityClassifier(LightningModule):
    def __init__(self, args):
        super().__init__()

        self.save_hyperparameters()

        self.args = args

        # initialize ResNet
        self.encoder = get_resnet(self.args.resnet, pretrained=self.args.pretrain)
        self.n_features = self.encoder.fc.in_features  # get dimensions of fc layer
        self.model = LightCnn(self.encoder, self.n_features, self.args.n_classes)
        self.test_outputs = np.array([])
        self.criterion = torch.nn.CrossEntropyLoss()

        self.training_accuracy_metric = torchmetrics.Accuracy()
        self.validation_accuracy_metric = torchmetrics.Accuracy()

    def forward(self, images):
        outputs = self.model(images)
        return outputs

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        images, labels = batch
        preds = self.forward(images)
        train_loss = self.criterion(preds, labels)

        self.log("Training Accuracy Batch", self.training_accuracy_metric(preds, labels), on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log("Training Loss", train_loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return train_loss

    def training_epoch_end(self, outputs):
        self.log("Training Accuracy Epoch", self.training_accuracy_metric.compute())
        self.training_accuracy_metric.reset()

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        preds = self.forward(images)
        val_loss = self.criterion(preds, labels)
        self.log("Validation Accuracy Batch", self.validation_accuracy_metric(preds, labels), on_step=True, on_epoch=True,
                 logger=True, prog_bar=True)
        self.log("Validation Loss", val_loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)

        return val_loss

    def validation_epoch_end(self, outputs):
        self.log("Validation Accuracy Epoch", self.validation_accuracy_metric.compute())
        self.validation_accuracy_metric.reset()

    def test_step(self, test_batch, batch_idx):
        images, labels = test_batch
        outputs = self.forward(images)
        outputs = outputs.cpu().numpy()

        if len(self.test_outputs) == 0:
            self.test_outputs = outputs
        else:
            self.test_outputs = np.append(self.test_outputs, outputs, axis=0)

    def test_epoch_end(self, outputs):
        output_csv = pd.DataFrame(self.test_outputs)
        output_csv.to_csv(self.args.csv_path + self.args.model_file[:-4] + "csv", header=False, index=False)

    # def configure_criterion(self):
    #     criterion = torch.nn.CrossEntropyLoss()
    #     return criterion

    def configure_optimizers(self):
        if self.args.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

            # "decay the learning rate with the cosine decay schedule without restarts"
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        else:
            raise NotImplementedError

        if self.args.scheduler:
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "Validation Loss_epoch"}
        else:
            return {"optimizer": optimizer}

# Helper function to show a batch
def show_batch(batch_images, batch_labels):
    """Show image with landmarks for a batch of samples."""
    batch_size = len(batch_images)
    im_size = batch_images.size(2)
    grid_border_size = 2

    grid = torchvision.utils.make_grid(batch_images)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    print()

if __name__ == "__main__":
    wandb_logger = WandbLogger(project="similarity classifiers", entity='unr-mpl')

    parser = argparse.ArgumentParser(description="similarity")
    contrastive_parser = argparse.ArgumentParser(description="contrastive")
    yaml_config = yaml_config_hook("./config.yaml")
    contrastive_learning_yaml_config = yaml_config_hook("/home/nthom/Documents/contrastive_learning/config/config.yaml")

    sweep = False
    if sweep:
        hyperparameter_defaults = dict(
            h_dim = 512,
            projection_dim = 64,
            temperature = 0.05,
            learning_rate = 0.0003,
        )

        wandb.init(config=hyperparameter_defaults)

        yaml_config = yaml_config_hook("./config.yaml")
        wandb.config.update(
            {k:v for k, v in yaml_config.items() if k not in wandb.config}
        )

        for k, v in wandb.config.items():
            parser.add_argument(f"--{k}", default=v, type=type(v))

        args = parser.parse_args()
    else:
        for k, v in yaml_config.items():
            parser.add_argument(f"--{k}", default=v, type=type(v))

        args = parser.parse_args()

    for k, v in contrastive_learning_yaml_config.items():
        contrastive_parser.add_argument(f"--{k}", default=v, type=type(v))
    contrastive_args = contrastive_parser.parse_args()

    pl.seed_everything(args.seed)

    if args.reload:
        if args.simclr_pretrained:
            cl = SimilarityClassifier(args)
            cl_dict = cl.state_dict()

            pretrained_model = ContrastiveLearning.load_from_checkpoint(args.model_path + args.model_file,
                                                                        args=contrastive_args)
            pretrained_model_dict = pretrained_model.state_dict()

            new_model_dict1 = {}
            new_model_dict2 = {}
            for pt_item_k, pt_item_v in pretrained_model_dict.items():
                for cl_item_k, cl_item_v in cl_dict.items():
                    if pt_item_v.shape == cl_item_v.shape:
                        new_model_dict1[cl_item_k] = cl_item_v

            cl_dict.update(new_model_dict1)
            cl.load_state_dict(cl_dict)

            args.image_size_h = cl.args.image_size_h
            args.image_size_w = cl.args.image_size_w
        else:
            cl = SimilarityClassifier.load_from_checkpoint(args.model_path + args.model_file, args=args)
    else:
        cl = SimilarityClassifier(args)

    if args.train:
        if args.dataset == "VGG_Face":
            train_dataset = vgg_face_dataset.Att_Dataset(
                args,
                fold="training",
                # transform=TransformsSimCLR(size=(args.image_size_h, args.image_size_w)),
                transform=None,
            )
            val_dataset = vgg_face_dataset.Att_Dataset(
                args,
                fold="validation",
                transform=None,
            )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, drop_last=False,
                                  shuffle=True, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers, drop_last=False,
                                  shuffle=False, persistent_workers=True)

        if args.show_batch:
            for sampled_batch_images, sampled_batch_labels in train_loader:
                plt.figure()
                show_batch(sampled_batch_images, sampled_batch_labels)
                plt.axis('off')
                plt.ioff()
                plt.show()

    elif args.test:
        if args.dataset == "VGG_Face":
            test_dataset = vgg_face_dataset.Att_Dataset(
                args,
                fold="testing",
                transform=None,
            )

        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers, drop_last=False,
                                 shuffle=False, persistent_workers=True)

        if args.show_batch:
            for sampled_batch_images, sampled_batch_labels in test_loader:
                plt.figure()
                show_batch(sampled_batch_images, sampled_batch_labels)
                plt.axis('off')
                plt.ioff()
                plt.show()
                # if i_batch == 10:
                #     print()

    if args.save == True:
        checkpoint_callback = ModelCheckpoint(
            monitor='Training Loss',
            dirpath=args.model_path,
            filename='{epoch:02d}-{Training Loss:.05f}-' + args.save_name,
            save_top_k=5,
            mode='min',
        )

        trainer = pl.Trainer(
            logger=wandb_logger,
            precision=16,
            callbacks=[checkpoint_callback],
            accelerator='ddp',
            plugins=DDPPlugin(find_unused_parameters=False),
            gpus=args.gpus,
            num_nodes=1,
            # limit_train_batches=0.01,
            # limit_val_batches=0.01,
            max_epochs=args.epochs
        )
    else:
        trainer = pl.Trainer(
            logger=wandb_logger,
            precision=16,
            checkpoint_callback=False,
            accelerator='ddp',
            plugins=DDPPlugin(find_unused_parameters=False),
            gpus=args.gpus,
            num_nodes=1,
            # limit_train_batches=0.01,
            max_epochs=args.epochs
        )

    if args.train == True:
        trainer.sync_batchnorm = True
        trainer.fit(cl, train_loader, val_loader)
        # trainer.fit(cl, train_loader)

    # if args.train == True:
    #     # trainer.fit(net, train_loader, val_loader)
    #     trainer.fit(cl, train_loader)
    #
    # if args.val_only == True:
    #     trainer.test(cl, val_loader)
    #
    elif args.test == True:
        trainer.test(cl, test_loader)