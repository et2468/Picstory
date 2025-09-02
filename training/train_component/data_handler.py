from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

class DataHandler:
    def __init__(self, data_dir, batch_size=16, num_workers=0):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.data_transforms = {
            "train": transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
        }

    def get_dataloaders(self):
        train_dir = os.path.join(self.data_dir, "training_set")
        val_dir = os.path.join(self.data_dir, "test_set")

        image_datasets = {
            "train": datasets.ImageFolder(train_dir, self.data_transforms["train"]),
            "val": datasets.ImageFolder(val_dir, self.data_transforms["val"]),
        }

        dataloaders = {
            "train": DataLoader(
                image_datasets["train"],
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            ),
            "val": DataLoader(
                image_datasets["val"],
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            ),
        }

        dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
        class_names = image_datasets["train"].classes

        return dataloaders, dataset_sizes, class_names