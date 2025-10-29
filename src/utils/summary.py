import logging, re
from torch import nn


def print_model_size(model: nn.Module):
    param_size = sum(p.numel() for p in model.parameters() if p.requires_grad) * 4  # 4 bytes per float32
    buffer_size = sum(p.numel() for p in model.buffers()) * 4  # 4 bytes per float32
    size_all_mb = (param_size + buffer_size) / (1024 ** 2)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total Params: {n_parameters:,}")
    logging.info(f"Model Params Size: {size_all_mb:.2f} MB")



def get_acronym(name: str) -> str:
    """
    Converts CamelCase or PascalCase string to an acronym
    (e.g., 'PeakSignalNoiseRatio' -> 'PSNR').
    Returns the original string if the acronym would be less than 2 letters.
    """
    letters = ''.join(re.findall(r'[A-Z]', name))
    return letters if len(letters) >= 2 else name
