"""
Exports a PyTorch model to onnx
"""
import logging
import torch, onnx
import os
from argparse import ArgumentParser
from src.utils.quantization import quantize_onnx_model
from src.utils.export import save_to_onnx
from src.utils.config import load_export_config_file, instanciate_module
import warnings
from src.utils.calibration import QuantizationDataReader
warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == "__main__":

    project_name = "medmnist-resnet"

    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - {project_name} Model Exportation - %(levelname)s - %(message)s'
    )

    parser = ArgumentParser()
    parser.add_argument("--export_config_path", type=str, required=True)
    args = parser.parse_args()

    export_config = load_export_config_file(args.export_config_path)
    export_path = os.path.join(
        export_config['export_path'], '{}.onnx'.format(export_config['export_name']))

    model_md = export_config['model']['module_name']
    model_cls = export_config['model']['class_name']
    model_params = export_config['model']['parameters']

    model: torch.nn.Module = instanciate_module(
        model_md,
        model_cls,
        model_params
    )
    model.load_state_dict(torch.load(
        export_config['model_path'], weights_only=True, map_location=torch.device('cpu'))['weights'])
    model.to('cpu')

    dataset = instanciate_module(
        export_config['dataset']['module_name'],
        export_config['dataset']['class_name'],
        export_config['dataset']['parameters']
    )

    quantization_dataset = QuantizationDataReader(dataset, 100)
    x = torch.Tensor(quantization_dataset.get_next()['input'])

    save_to_onnx(export_path, model, x)

    logging.info('Running model quantization ...')

    quantized_model_path = os.path.join(
        export_config['export_path'], '{}_quantized.onnx'.format(export_config['export_name']))
    quantize_onnx_model(export_path, quantized_model_path,
                        calibration_dataset=quantization_dataset)
    logging.info('Exportation done ✅')
