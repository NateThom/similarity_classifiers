import torch
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision

class Att_Dataset(Dataset):
    def __init__(self, args, fold, transform=None):
        if fold == "training":
            lower_bound = 0
            upper_bound = args.train_size
            # upper_bound = 107304
            # upper_bound = 59971
            # upper_bound = 1154
        elif fold == "validation":
            lower_bound = args.train_size
            upper_bound = (args.train_size + args.val_size)
        elif fold == "testing":
            lower_bound = (args.train_size + args.val_size)
            upper_bound = (args.train_size + args.val_size + args.test_size)
        elif fold == "not test":
            lower_bound = 0
            upper_bound = (args.train_size + args.val_size)
        else:
            lower_bound = 0
            upper_bound = args.all_size

        self.train = args.train
        self.test = args.test

        self.img_path = args.image_path
        self.img_dir = args.image_dir

        # Read the binary attribute labels from the specified file
        self.img_labels = pd.read_csv(args.label_path, sep=',', skiprows=0, usecols=[1])[lower_bound:upper_bound]

        # Get the paths to each of the input images
        self.input_filenames = pd.read_csv(args.label_path, sep=',', skiprows=0, usecols=[0])[lower_bound:upper_bound]

        # If there are any transform functions to be called, store them
        self.transform = transform

    def __len__(self):
        return len(self.input_filenames)

    def __getitem__(self, idx):
        img_path = self.input_filenames.iloc[idx, 0]
        # img_path = self.img_path + self.img_dir + img_path[55:]
        image = torchvision.io.read_image(img_path)
        image = TF.convert_image_dtype(image, torch.float)

        # Read in the attribute labels for the current input image
        labels = self.img_labels.iloc[idx,]
        labels = torch.squeeze(torch.tensor(labels))

        if image.shape[0] != 3:
            image = image.repeat(3, 1, 1)

        if self.transform:
            image = self.transform(image)

        return image, labels