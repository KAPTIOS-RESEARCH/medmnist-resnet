import torch
import os
import logging
import onnx
from .quantization import fuse_layers
import onnxoptimizer as optimizer

def save_to_onnx(path: str, model: torch.nn.Module, tensor_x: torch.Tensor):
    """Exports a PyTorch model to ONNX format.

    Args:
        path (str): The directory were the .onnx file will be saved
        model (torch.nn.Module): The model to export
        tensor_x (torch.Tensor): A data sample used to train the model
    """
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)

    model = fuse_layers(model)
    onnx_program = torch.onnx.export(
        model,
        (tensor_x,),
        dynamo=True,
        opset_version=21,
        input_names=['input'],
        output_names=['output'],
    )
    
    onnx_program.save(path)
    logging.info(f'Model successfully exported to ONNX format in {path}')
    
    try:
        onnx_model = onnx.load(path)
        onnx.checker.check_model(onnx_model)
        logging.info('ONNX model is valid.')
    except onnx.onnx_cpp2py_export.checker.ValidationError as e:
        logging.error(f'ONNX model validation failed: {e}')
        raise
    
    optimize_onnx_model(path)
    logging.info(f'ONNX Model successfully optimized and saved to {path}')


def optimize_onnx_model(model_path: str):
    """Optimize an ONNX model by removing unused nodes and fusing layers.

    Args:
        model_path (str): Path to the original ONNX model file
    """

    model = onnx.load(model_path)
    passes = ['extract_constant_to_initializer',
              'eliminate_unused_initializer']
    
    try:
        optimized_model = optimizer.optimize(model, passes)
        onnx.save(optimized_model, model_path)
    except Exception as e:
        logging.warning(f'Skipping ONNX optimization due to error: {e}')
        