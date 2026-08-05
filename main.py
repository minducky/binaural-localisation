"""Entrypoint: runs the experiment sweep defined by a YAML config file."""

import argparse

from experiment_manager import ExperimentManager

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    manager = ExperimentManager(config_fpath=args.config)
    manager.run_all_experiments()
