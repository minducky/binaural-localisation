"""Loads a sweep config and runs each experiment: train, evaluate, log to wandb."""

import gc
import os
from copy import deepcopy
from datetime import datetime
from time import time

import torch
import wandb
import yaml

from dataset import setup_develop_dataloaders, setup_eval_dataloaders
from models.get_model import get_model
from result_plotter import ResultPlotter
from trainer import Trainer


class ExperimentManager:
    """Runs a sweep of experiments defined by a YAML config's parallel lists."""

    def __init__(self, config_fpath: str = "config.yaml"):
        """Loads the sweep config and creates the results base directory.

        Args:
            config_fpath: Path to the YAML config file.
        """
        with open(config_fpath) as f:
            self.config = yaml.safe_load(f)  # Read config.yaml file as config
        self.exps_purpose = self.config[
            "EXPERIMENTS_PURPOSE"
        ]  # Purpose of Experiment (Also used as wandb group)
        self.exp_configs = self.config[
            "EXPERIMENT_CONFIGS"
        ]  # When multiple experiments, each of experiment config as a list
        self.cache_size = self.config[
            "CACHE_SIZE"
        ]  # How many hdf5 files will you send to RAM memory
        self._validate_experiments()

        # Setup results directory
        self.exps_base_dir = os.path.join(
            self.config["RESULTS_DIR"], f"{self.exps_purpose}"
        )
        os.makedirs(self.exps_base_dir, exist_ok=True)

    # %% Setting Experiments
    def _validate_experiments(self) -> None:
        """Ensure all experiment config lists have same length"""
        lengths = {key: len(value) for key, value in self.exp_configs.items()}
        unique_lengths = set(lengths.values())
        assert (
            len(unique_lengths) == 1
        ), f"All keys must have same length. Found: {lengths}"
        self.num_experiments = list(lengths.values())[0]

    def get_experiment_config(self, exp_idx: int) -> dict:
        """Get config for specific experiment"""
        exp_config = deepcopy(self.config)
        for key, values in self.exp_configs.items():
            exp_config[key] = values[exp_idx]
        return exp_config

    def create_experiment_dir(self, exp_config: dict, exp_idx: int) -> dict:
        """Create directory structure for experiment"""
        exp_name = f'exp{exp_idx}_{exp_config["DESCRIPTION"]}'
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        exp_dir = os.path.join(
            self.exps_base_dir, f'{timestamp}_{exp_name}_gpu{exp_config["GPU_NUM"]}'
        )

        dirs = {
            "root": exp_dir,
            "checkpoint": os.path.join(exp_dir, "checkpoint"),
            "model": os.path.join(exp_dir, "model"),
            "history": os.path.join(exp_dir, "history"),
            "eval": os.path.join(exp_dir, "eval"),
        }
        for path in dirs.values():
            os.makedirs(path, exist_ok=True)

        return dirs

    def save_experiment_config(self, exp_config: dict, exp_dirs: dict) -> None:
        """Save experiment configuration to yaml"""
        config_path = os.path.join(exp_dirs["root"], "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(exp_config, f, default_flow_style=False, sort_keys=False)
        print(f"\n{'=' * 70}")
        print(f"Config saved to '{config_path}'")
        print(f"{'=' * 70}")

    # %% Initialisation of WandB
    def init_wandb(self, exp_config: dict, exp_idx: int, exp_dirs: dict) -> None:
        """Initialize Weight & Biases"""
        wandb.init(
            project="binaural-localisation",
            group=self.exps_purpose,
            name=f'exp{exp_idx}_{exp_config["DESCRIPTION"]}',
            config=exp_config,
            mode="offline",
            dir=exp_dirs["root"],
            reinit=True,
            settings=wandb.Settings(
                _disable_stats=True, _disable_meta=True, console="off"
            ),
        )

    # %% Running Experiments
    def run_experiment(self, exp_idx: int) -> float:
        """Run single experiment"""
        print(f"\n{'='*70}")
        print(f"Experiment {exp_idx+1} / {self.num_experiments}")
        print(f"{'='*70}")

        # Get experiment config
        exp_config = self.get_experiment_config(exp_idx)

        print(f"\n{'=' * 70}")
        print("Experiment configuration:")
        for key in self.exp_configs.keys():
            print(f" - {key}: {exp_config[key]}")
        print(f"{'=' * 70}")

        # Apply debug config

        if exp_config["DEBUG"]:
            print(f"\n{'-' * 70}")
            print("🔧 DEBUG MODE ENABLED")
            debug_config = exp_config["DEBUG_CONFIG"]
            exp_config.update(debug_config)
            print(f"{'-' * 70}")

        exp_start = time()

        # Setup directories
        exp_dirs = self.create_experiment_dir(exp_config, exp_idx)

        # Save experiment config
        self.save_experiment_config(exp_config, exp_dirs)

        # Create plotter
        plotter = ResultPlotter(exp_dirs)

        # Setup dataloaders and get global class mapping
        develop_dataloaders, global_class_mapping = setup_develop_dataloaders(
            exp_config
        )  # ✨ MODIFIED

        # Auto-set NUM_CLASSES from actual class mapping (driven by DATASET_MODE)
        exp_config["NUM_CLASSES"] = len(global_class_mapping["class_to_angle"])

        dataset_mode = exp_config.get("DATASET_MODE", "Total")
        print(f"\n{'=' * 70}")
        print(
            f" - NUM_CLASSES auto-set to {exp_config['NUM_CLASSES']} "
            f"(DATASET_MODE: {dataset_mode})"
        )
        print(f"{'=' * 70}")
        # Initialise wandb
        self.init_wandb(exp_config, exp_idx, exp_dirs)

        # Create model
        model = get_model(exp_config)

        # Log model params to wandb config
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Train
        trainer = Trainer(model, exp_config, develop_dataloaders, exp_dirs, plotter)
        train_start = time()
        train_results = trainer.train()
        train_elapsed = time() - train_start

        del develop_dataloaders
        trainer.dataloaders = None
        gc.collect()
        # Evaluate with same class mapping as training
        eval_dataloader = setup_eval_dataloaders(exp_config, global_class_mapping)
        eval_results = trainer.evaluate(eval_dataloader)

        # Log eval results
        print(f' - eval Loss: {eval_results["eval_loss"]:.4f}')
        print(f' - eval Accuracy: {eval_results["eval_accuracy"]:.4f}')

        # Log metrics to wandb
        wandb.log(
            {
                "total_params": total_params,
                "trainable_params": trainable_params,  # parameters
                "training_time_sec": train_elapsed,
                "avg_epoch_time_sec": train_results[
                    "avg_epoch_time"
                ],  # training time / avg epoch time
                "best_val_loss": train_results["best_val_loss"],  # best validation loss
                "peak_gpu_memory_gb": (
                    torch.cuda.max_memory_allocated() / 1e9
                    if torch.cuda.is_available()
                    else 0
                ),  # gpu allocated memory
                "eval_loss": eval_results["eval_loss"],  # evaluation loss
                "eval_accuracy": eval_results[
                    "eval_accuracy"
                ],  # eval combined accuracy
                "azimuth_accuracy": eval_results["overall_results"][
                    "azimuth_accuracy"
                ],  # eval azimuth accuracy
                "elevation_accuracy": eval_results["overall_results"][
                    "elevation_accuracy"
                ],  # eval elevation accuracy
                "spherical_error": eval_results["overall_results"][
                    "spherical_error"
                ],  # eval spherical error
                "azimuth_error": eval_results["overall_results"][
                    "azimuth_error"
                ],  # eval azimuth error
                "elevation_error": eval_results["overall_results"][
                    "elevation_error"
                ],  # eval elevation error
            }
        )

        # Log SNR-wise eval results as line plots
        metric_names = [
            "spherical_error",
            "azimuth_error",
            "elevation_error",
            "azimuth_accuracy",
            "elevation_accuracy",
            "combined_accuracy",
        ]
        for metric_name in metric_names:
            table = wandb.Table(
                data=[
                    [float(snr_key.replace("SNR_", "")), metrics[metric_name]]
                    for snr_key, metrics in eval_results["results_by_snr"].items()
                ],
                columns=["snr", metric_name],
            )
            wandb.log(
                {
                    f"eval_by_snr/{metric_name}": wandb.plot.line(
                        table, "snr", metric_name, title=f"{metric_name} by SNR"
                    )
                }
            )

        wandb.finish()

        exp_elapsed = time() - exp_start
        print(
            f"\nExperiment {exp_idx + 1} completed in {self.format_time(exp_elapsed)}"
        )

        return exp_elapsed

    def run_all_experiments(self) -> None:
        """Run all experiments"""
        total_start = time()

        print(f"\n{'='*70}")
        print(f"Experiments Purpose: {self.exps_purpose}")
        print(f"Number of Experiments: {self.num_experiments}")
        print(f"{'='*70}\n")

        exp_times = []
        for exp_idx in range(self.num_experiments):
            elapsed = self.run_experiment(exp_idx)
            exp_times.append(elapsed)

        # Summary
        total_elapsed = time() - total_start
        print(f"\n{'='*70}")
        print("ALL EXPERIMENTS COMPLETED")
        print(f"{'='*70}")
        print(f"Total time: {self.format_time(total_elapsed)}")
        for i, t in enumerate(exp_times):
            print(f"  - Experiment {i}: {self.format_time(t)}")
        print("\nCheers!!")

    # %% Just for Formatting Time (s -> h, m, s)
    @staticmethod
    def format_time(seconds: float) -> str:
        """Format elapsed time"""
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{int(hours)}h {int(minutes)}m {int(secs)}s"
