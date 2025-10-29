import torchvision
from onnxruntime.quantization import CalibrationDataReader
from torchvision.datasets import MNIST
import numpy as np


class MNISTCalibrationDataset(CalibrationDataReader):
    def __init__(self, data_dir: str,
                 num_samples: int = 100):
        self.num_samples = num_samples
        self.enum_data_dicts = None
        self.load_data(data_dir)

    def load_data(self, data_dir: str):

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = MNIST(
            root=data_dir,
            train=False,
            transform=transform,
            download=True
        )
        self.data_dicts = []

        for i in range(self.num_samples):
            x, _ = dataset[i]
            x = x.numpy()
            x = np.expand_dims(x, axis=0)
            self.data_dicts.append({'input': x.astype(np.float32)})

        self.enum_data_dicts = iter(self.data_dicts)

    def get_next(self):
        return next(self.enum_data_dicts, None)

    def rewind(self):
        self.enum_data_dicts = iter(self.data_dicts)
