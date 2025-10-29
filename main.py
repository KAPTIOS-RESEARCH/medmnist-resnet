import warnings
warnings.filterwarnings("ignore")
import logging, os, torch, wandb
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from datetime import datetime
from argparse import ArgumentParser
from src.utils.config import load_config_file, instanciate_module
from src.core.experiment import AbstractExperiment
from src.utils.energy.sampler import EnergySampler

if __name__ == "__main__":

    project_name = "medmnist-resnet"

    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - {project_name} Model Training - %(levelname)s - %(message)s'
    )

    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()
    config = load_config_file(args.config_path)
    
    experiment_cls = config['experiment']['class_name']
    experiment_md = config['experiment']['module_name']
    experiment: AbstractExperiment = instanciate_module(
        experiment_md,
        experiment_cls,
        {"config": config}
    )
    
    if config['track']:
        now = datetime.now()
        date_time = now.strftime("%Y_%m_%d_%H_%M_%S_%f")
        wandb.init(project=project_name,
            name=experiment.name, 
            config=config,
            id=date_time,
            dir=experiment.log_dir
        )
        wandb.watch(experiment.model)

    energy_sampler = EnergySampler(
        cpu_tdp_w=45,
        dram_w_per_gb=1.5,
        interval_s=config['energy_sampling_interval'],
        log_to_wandb=config['track']
    )
    
    energy_sampler.start()
    experiment.run()
    energy_sampler.stop()
    energy_metrics = {
        "energy/avg_cpu_power_w": energy_sampler.avg_cpu_power_w,
        "energy/avg_dram_power_w": energy_sampler.avg_dram_power_w,
    }
    model_file_path = os.path.join(experiment.log_dir, 'best_model.pth')
    logging.info(f'⚡ Saving energy consumption metrics to {model_file_path}')
    model_object = torch.load(model_file_path, map_location=torch.device('cpu'))
    model_object['energy_consumption'] = energy_metrics
    torch.save(model_object, model_file_path)
    logging.info(f"✅ Updated model saved with energy_consumption at {model_file_path}")

    if config['track']:
        wandb.finish()