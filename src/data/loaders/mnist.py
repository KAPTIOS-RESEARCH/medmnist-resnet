from torch.utils.data import Subset, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST


class MNISTDataloader(object):
    def __init__(self,
                 data_dir: str,
                 batch_size: int = 4,
                 num_workers: int = 4,
                 debug: bool = True):

        super(MNISTDataloader, self).__init__()
        self.data_dir = data_dir
        self.debug = debug
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = Compose([
            ToTensor(),
            Normalize((0.1307,), (0.3081,))
        ])

    def train(self):
        train_dataset = MNIST(
            root=self.data_dir,
            train=True,
            transform=self.transform,
            download=True
        )

        if self.debug:
            train_dataset = Subset(train_dataset, range(self.batch_size * 2))

        dataloader = DataLoader(train_dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=True,
                                pin_memory=True)
        return dataloader

    def val(self):
        val_dataset = MNIST(
            root=self.data_dir,
            train=False,
            transform=self.transform,
            download=True
        )

        if self.debug:
            val_dataset = Subset(val_dataset, range(self.batch_size * 2))

        dataloader = DataLoader(val_dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=False,
                                pin_memory=True)
        return dataloader

    def test(self):
        return self.val()
