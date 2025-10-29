import torchvision.transforms as transforms
import numpy as np
from onnxruntime.quantization import CalibrationDataReader


class QuantizationDataReader(CalibrationDataReader):
    def __init__(self, torch_ds, num_samples: int = 100):

        self.torch_ds = torch_ds
        self.torch_ds.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.num_samples = num_samples
        self.enum_data_dicts = None
        self.data_dicts = []

        for i in range(self.num_samples):
            x, _ = self.torch_ds[i]
            x = x.numpy()
            x = np.expand_dims(x, axis=0)
            self.data_dicts.append({'input': x.astype(np.float32)})

        self.enum_data_dicts = iter(self.data_dicts)

    def get_next(self):
        return next(self.enum_data_dicts, None)

    def rewind(self):
        self.enum_data_dicts = iter(self.data_dicts)
