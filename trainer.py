"""Train/val loop, early stopping, checkpointing, and evaluation metrics."""

import math
import os
from time import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from result_plotter import ResultPlotter


class Trainer:
    """Owns the model, optimizer, and scheduler for one experiment run."""

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        dataloaders: dict,
        exp_dirs: dict,
        plotter: ResultPlotter,
    ):
        """Sets up the model, optimizer, LR schedule, and loss for training.

        Args:
            model: The model to train.
            config: Experiment config dict (learning rate, epochs, etc.).
            dataloaders: Dict with 'train'/'val' ``DataLoader``s.
            exp_dirs: Dict of experiment output directories (from
                ``ExperimentManager.create_experiment_dir``).
            plotter: ``ResultPlotter`` used to save evaluation figures.
        """
        self.model = model
        self.config = config
        self.dataloaders = dataloaders
        self.exp_dirs = exp_dirs
        self.plotter = plotter

        # Device
        self.device = torch.device(
            f'cuda:{config["GPU_NUM"]}' if torch.cuda.is_available() else "cpu"
        )
        print(f"device : {self.device}")
        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config["LEARNING_RATE"],
            weight_decay=config["WEIGHT_DECAY"],
        )

        # Scheduler: linear warmup → cosine annealing
        warmup_epochs = config["WARMUP_EPOCHS"]
        num_epochs = config["NUM_EPOCHS"]

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Training params
        self.num_epochs = num_epochs
        self.patience = config["PATIENCE"]

        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.bm_fc_history = []
        self.bm_ourbeta_history = []

    def train_epoch(self, enable_profiler: bool = False) -> float | None:
        """Runs one training epoch.

        Args:
            enable_profiler: If True, profiles a few batches with
                ``torch.profiler`` and prints a summary instead of returning
                a loss.

        Returns:
            Mean training loss for the epoch, or None if ``enable_profiler``
            was True.
        """
        self.model.train()
        epoch_loss = 0

        pbar = tqdm(
            self.dataloaders["train"],
            desc="Training",
            leave=True,
            ncols=100,
            colour="blue",
        )

        if enable_profiler:
            prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=1, warmup=3, active=5),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    os.path.join(self.exp_dirs["root"], "profiler")
                ),
                record_shapes=True,
                with_flops=True,
                with_stack=True,
            )
            prof.start()

        for batch_idx, (x, y, _, _, _) in enumerate(pbar):
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if enable_profiler:
                prof.step()
                if batch_idx >= 8:  # wait(1)+warmup(1)+active(3) = 5 steps
                    break

        if enable_profiler:
            prof.stop()
            print("\n[Profiler] Top operations by CUDA time:")
            print(
                prof.key_averages(group_by_stack_n=5).table(
                    sort_by="cuda_time_total", row_limit=50
                )
            )
            # events = sorted(prof.events(), key=lambda e: e.time_range.start)
            # for e in events:
            #     if e.cuda_time_total > 0:
            #         print(f"{e.name:60s}  cuda={e.cuda_time_total / 1e3:8.2f}ms")

            return None  # profiler run은 loss 반환 안 함

        return epoch_loss / len(self.dataloaders["train"])

    def validate_epoch(self) -> float:
        """Runs one validation epoch and returns the mean loss."""
        self.model.eval()

        epoch_loss = 0

        with torch.no_grad():
            pbar = tqdm(
                self.dataloaders["val"],
                desc="Validation",
                leave=True,
                ncols=100,
                colour="magenta",
            )
            for x, y, _, _, _ in pbar:
                x, y = x.to(self.device), y.to(self.device)

                outputs = self.model(x)
                loss = self.criterion(outputs, y)

                epoch_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return epoch_loss / len(self.dataloaders["val"])

    def train(self) -> dict:
        """Runs the main training loop with early stopping.

        Returns:
            A dict with ``best_val_loss``, ``stopped_epoch``, and
            ``avg_epoch_time``.
        """
        epoch_pbar = tqdm(
            range(1, self.num_epochs + 1), desc="Epochs", ncols=120, colour="cyan"
        )

        epoch_times = []
        stopped_epoch = self.num_epochs

        if self.config["PROFILER"]:
            self.train_epoch(enable_profiler=True)
            torch.cuda.empty_cache()

        for epoch in epoch_pbar:
            epoch_start = time()
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()
            epoch_times.append(time() - epoch_start)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Update progress bar
            epoch_pbar.set_postfix(
                {"train_loss": f"{train_loss:.4f}", "val_loss": f"{val_loss:.4f}"}
            )

            # Wandb logging
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                }
            )

            # BM learnable parameter tracking
            bm_params = self._get_bm_params()
            if bm_params is not None:
                self.bm_fc_history.append(bm_params["fc"])
                self.bm_ourbeta_history.append(bm_params["ourbeta"])

            self.scheduler.step()

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint()
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                stopped_epoch = epoch
                self._load_checkpoint()
                break

            torch.cuda.empty_cache()

        self._save_final_model()
        self._save_losses()
        self._save_bm_history()

        return {
            "best_val_loss": self.best_val_loss,
            "stopped_epoch": stopped_epoch,
            "avg_epoch_time": sum(epoch_times) / len(epoch_times),
        }

    def evaluate(self, eval_dataloader: dict) -> dict:
        """Evaluates on the eval set: loss, angular error/accuracy, per-SNR breakdown.

        Args:
            eval_dataloader: Dict with an ``'eval'`` ``DataLoader`` (as
                returned by ``setup_eval_dataloaders``).

        Returns:
            A dict with ``eval_loss``, ``eval_accuracy``, ``overall_results``,
            and ``results_by_snr``.
        """
        self.model.eval()
        eval_loss = 0

        # Collect predictions and ground truth
        all_preds = []
        all_labels = []
        all_pred_azims = []
        all_pred_elevs = []
        all_true_azims = []
        all_true_elevs = []
        all_snrs = []

        # Get class_to_angle mapping from dataset
        dataset = eval_dataloader["eval"].dataset

        # Handle Subset wrapper if in debug mode
        if hasattr(dataset, "dataset"):
            base_dataset = dataset.dataset
        else:
            base_dataset = dataset
        class_to_angle = base_dataset.class_to_angle

        # Build prior mask: 1.0 for classes present in eval set, 0.0 otherwise
        n_classes = len(class_to_angle)
        prior_np = base_dataset.build_eval_prior(n_classes)
        prior = torch.tensor(prior_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            pbar = tqdm(
                eval_dataloader["eval"], desc="Evaluating", ncols=100, colour="green"
            )
            for (
                x,
                y,
                _azim,
                _elev,
                snr,
            ) in pbar:  # now includes elev and snr (only when evaluation)
                x, y = x.to(self.device), y.to(self.device)

                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                eval_loss += loss.item()

                probs = torch.softmax(outputs, dim=1)  # (B, n_classes)
                masked = probs * prior.unsqueeze(0)  # zero out non-eval classes
                preds = masked.argmax(dim=1).cpu().numpy()

                labels = y.cpu().numpy()

                # Convert class indices to angles
                for pred_cls, true_cls in zip(preds, labels, strict=True):
                    pred_azim, pred_elev = class_to_angle[pred_cls]
                    true_azim, true_elev = class_to_angle[true_cls]

                    all_pred_azims.append(pred_azim)
                    all_pred_elevs.append(pred_elev)
                    all_true_azims.append(true_azim)
                    all_true_elevs.append(true_elev)

                all_preds.extend(preds)
                all_labels.extend(labels)
                all_snrs.extend(snr.numpy())

        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_pred_azims = np.array(all_pred_azims)
        all_pred_elevs = np.array(all_pred_elevs)
        all_true_azims = np.array(all_true_azims)
        all_true_elevs = np.array(all_true_elevs)
        all_snrs = np.array(all_snrs)
        # Get unique azimuths and elevations in evaluation set
        unique_eval_azims = np.unique(all_true_azims)
        unique_eval_elevs = np.unique(all_true_elevs)
        unique_eval_classes = np.unique(all_labels)

        print(f"\n{'='*70}")
        print("Evaluation Dataset Statistics")
        print(f"{'='*70}")
        n_train_azims = len(base_dataset.azim_values)
        n_train_elevs = len(base_dataset.elev_values)
        n_train_classes = len(class_to_angle)
        print(f"  Total samples: {len(all_labels)}")
        print(
            f"  Unique azimuths: {len(unique_eval_azims)} "
            f"(out of {n_train_azims} in training)"
        )
        print(
            f"  Azimuth range: {unique_eval_azims.min():.1f}° to "
            f"{unique_eval_azims.max():.1f}°"
        )
        print(
            f"  Unique elevations: {len(unique_eval_elevs)} "
            f"(out of {n_train_elevs} in training)"
        )
        print(
            f"  Elevation range: {unique_eval_elevs.min():.1f}° to "
            f"{unique_eval_elevs.max():.1f}°"
        )
        print(
            f"  Unique classes: {len(unique_eval_classes)} "
            f"(out of {n_train_classes} in training)"
        )
        print(f"  SNR range: {all_snrs.min():.1f} dB to {all_snrs.max():.1f} dB")

        # Calculate metrics by SNR
        unique_snrs = np.unique(all_snrs)
        results_by_snr = {}

        print(f"\n{'='*70}")
        print("Evaluation Results by SNR")
        print(f"{'='*70}")

        for snr_val in sorted(unique_snrs):
            mask = all_snrs == snr_val
            snr_results = self._calculate_metrics(
                all_pred_azims[mask],
                all_pred_elevs[mask],
                all_true_azims[mask],
                all_true_elevs[mask],
                all_preds[mask],
                all_labels[mask],
            )
            results_by_snr[f"SNR_{snr_val}"] = snr_results

            print(f"\nSNR = {snr_val} dB ({mask.sum()} samples):")
            print(f"  Spherical Error: {snr_results['spherical_error']:.2f}°")
            print(f"  Azimuth Error: {snr_results['azimuth_error']:.2f}°")
            print(f"  Elevation Error: {snr_results['elevation_error']:.2f}°")
            print(f"  Azimuth Accuracy: {snr_results['azimuth_accuracy']:.2%}")
            print(f"  Elevation Accuracy: {snr_results['elevation_accuracy']:.2%}")
            print(f"  Combined Accuracy: {snr_results['combined_accuracy']:.2%}")

        # Overall metrics (all SNRs combined)
        overall_results = self._calculate_metrics(
            all_pred_azims,
            all_pred_elevs,
            all_true_azims,
            all_true_elevs,
            all_preds,
            all_labels,
        )

        print(f"\n{'='*70}")
        print("Overall Results (All SNRs)")
        print(f"{'='*70}")
        print(f"  Spherical Error: {overall_results['spherical_error']:.2f}°")
        print(f"  Azimuth Error: {overall_results['azimuth_error']:.2f}°")
        print(f"  Elevation Error: {overall_results['elevation_error']:.2f}°")
        print(f"  Azimuth Accuracy: {overall_results['azimuth_accuracy']:.2%}")
        print(f"  Elevation Accuracy: {overall_results['elevation_accuracy']:.2%}")
        print(f"  Combined Accuracy: {overall_results['combined_accuracy']:.2%}")
        print(f"{'='*70}\n")

        # Prepare evaluation metadata
        eval_metadata = {
            "total_samples": len(all_labels),
            "num_unique_azims": len(unique_eval_azims),
            "num_unique_elevs": len(unique_eval_elevs),
            "num_unique_classes": len(unique_eval_classes),
            "azim_range": (
                float(unique_eval_azims.min()),
                float(unique_eval_azims.max()),
            ),
            "elev_range": (
                float(unique_eval_elevs.min()),
                float(unique_eval_elevs.max()),
            ),
            "snr_range": (float(all_snrs.min()), float(all_snrs.max())),
            "unique_azims": unique_eval_azims.tolist(),
            "unique_elevs": unique_eval_elevs.tolist(),
        }

        # Build confusion matrices
        conf_matrices = self._build_confusion_matrices(
            all_true_azims,
            all_pred_azims,
            all_true_elevs,
            all_pred_elevs,
            all_labels,
            all_preds,
            class_to_angle,
        )

        # Plot results
        eval_dir = self.exp_dirs["eval"]
        os.makedirs(eval_dir, exist_ok=True)
        self.plotter.plot_confusion_matrices(conf_matrices, eval_dir)

        # Save detailed results to CSV and probability outputs
        self._save_evaluation_results(
            results_by_snr,
            overall_results,
            eval_loss,
            len(eval_dataloader["eval"]),
            eval_metadata,
            conf_matrices,
        )

        results = {
            "eval_loss": eval_loss / len(eval_dataloader["eval"]),
            "eval_accuracy": overall_results["combined_accuracy"],
            "results_by_snr": results_by_snr,
            "overall_results": overall_results,
            "eval_metadata": eval_metadata,
        }

        return results

    def _calculate_metrics(
        self,
        pred_azims: np.ndarray,
        pred_elevs: np.ndarray,
        true_azims: np.ndarray,
        true_elevs: np.ndarray,
        pred_classes: np.ndarray,
        true_classes: np.ndarray,
    ) -> dict:
        """Calculates angular errors and classification accuracies.

        Returns:
            A dict with ``spherical_error``, ``azimuth_error``,
            ``elevation_error``, ``azimuth_accuracy``, ``elevation_accuracy``,
            and ``combined_accuracy``.
        """
        # Angular errors
        azim_errors = self._angular_difference(pred_azims, true_azims)
        elev_errors = np.abs(pred_elevs - true_elevs)

        # Spherical error (great circle distance)
        spherical_errors = self._spherical_distance(
            pred_azims, pred_elevs, true_azims, true_elevs
        )

        # Classification accuracies
        azim_correct = pred_azims == true_azims
        elev_correct = pred_elevs == true_elevs
        combined_correct = (azim_correct) & (elev_correct)

        return {
            "spherical_error": np.mean(spherical_errors),
            "azimuth_error": np.mean(azim_errors),
            "elevation_error": np.mean(elev_errors),
            "azimuth_accuracy": np.mean(azim_correct),
            "elevation_accuracy": np.mean(elev_correct),
            "combined_accuracy": np.mean(combined_correct),
        }

    def _angular_difference(self, angle1: np.ndarray, angle2: np.ndarray) -> np.ndarray:
        """Calculate minimum angular difference between two angles"""
        diff = np.abs(angle1 - angle2)
        # Handle wraparound
        diff = np.minimum(diff, 360 - diff)
        return diff

    def _spherical_distance(
        self,
        azim1: np.ndarray,
        elev1: np.ndarray,
        azim2: np.ndarray,
        elev2: np.ndarray,
    ) -> np.ndarray:
        """Calculate great circle distance on sphere"""
        # Convert to radians
        azim1_rad = np.deg2rad(azim1)
        elev1_rad = np.deg2rad(elev1)
        azim2_rad = np.deg2rad(azim2)
        elev2_rad = np.deg2rad(elev2)

        # Haversine formula for great circle distance
        dlat = elev2_rad - elev1_rad
        dlon = azim2_rad - azim1_rad

        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(elev1_rad) * np.cos(elev2_rad) * np.sin(dlon / 2) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(a))

        return np.rad2deg(c)

    def _build_confusion_matrices(
        self,
        true_azims: np.ndarray,
        pred_azims: np.ndarray,
        true_elevs: np.ndarray,
        pred_elevs: np.ndarray,
        true_classes: np.ndarray,
        pred_classes: np.ndarray,
        class_to_angle: dict,
    ) -> dict:
        """Builds azimuth/elevation/class confusion matrices for plotting.

        Returns:
            A dict with ``cm_azim``/``cm_elev``/``cm_class`` matrices and
            their corresponding tick-label lists.
        """

        def _to_signed(a):
            return a if a <= 180 else a - 360

        # Azimuth: collapse elevation; convert 0-360 → -180~180, sort normally
        true_azims_s = np.array([_to_signed(a) for a in true_azims])
        pred_azims_s = np.array([_to_signed(a) for a in pred_azims])
        azim_labels = sorted(np.unique(true_azims_s).tolist())
        cm_azim = confusion_matrix(true_azims_s, pred_azims_s, labels=azim_labels)

        # Elevation: collapse azimuth
        elev_labels = sorted(np.unique(true_elevs).tolist())
        cm_elev = confusion_matrix(true_elevs, pred_elevs, labels=elev_labels)

        # Class: sort by (elev, signed_azim) for block structure
        sorted_classes = sorted(
            np.unique(true_classes).tolist(),
            key=lambda c: (class_to_angle[c][1], _to_signed(class_to_angle[c][0])),
        )
        cm_class = confusion_matrix(true_classes, pred_classes, labels=sorted_classes)
        class_tick_labels = [
            f"e{class_to_angle[c][1]:.0f}/a{_to_signed(class_to_angle[c][0]):.0f}"
            for c in sorted_classes
        ]

        return {
            "cm_azim": cm_azim,
            "azim_labels": [str(int(a)) for a in azim_labels],
            "cm_elev": cm_elev,
            "elev_labels": elev_labels,
            "cm_class": cm_class,
            "class_labels": class_tick_labels,
        }

    def _save_evaluation_results(
        self,
        results_by_snr: dict,
        overall_results: dict,
        eval_loss: float,
        num_batches: int,
        eval_metadata: dict,
        conf_matrices: dict,
    ) -> None:
        """Save evaluation results to CSV files"""
        eval_dir = self.exp_dirs["eval"]

        metadata_df = pd.DataFrame(
            [
                {
                    "total_samples": eval_metadata["total_samples"],
                    "num_unique_azims": eval_metadata["num_unique_azims"],
                    "num_unique_elevs": eval_metadata["num_unique_elevs"],
                    "num_unique_classes": eval_metadata["num_unique_classes"],
                    "azim_min": eval_metadata["azim_range"][0],
                    "azim_max": eval_metadata["azim_range"][1],
                    "elev_min": eval_metadata["elev_range"][0],
                    "elev_max": eval_metadata["elev_range"][1],
                    "snr_min": eval_metadata["snr_range"][0],
                    "snr_max": eval_metadata["snr_range"][1],
                }
            ]
        )
        metadata_df.to_csv(os.path.join(eval_dir, "eval_metadata.csv"), index=False)

        # Save SNR-wise results
        snr_data = []
        for snr_key, metrics in results_by_snr.items():
            snr_val = snr_key.replace("SNR_", "")
            snr_data.append(
                {
                    "snr": snr_val,
                    "spherical_error": metrics["spherical_error"],
                    "azimuth_error": metrics["azimuth_error"],
                    "elevation_error": metrics["elevation_error"],
                    "azimuth_accuracy": metrics["azimuth_accuracy"],
                    "elevation_accuracy": metrics["elevation_accuracy"],
                    "combined_accuracy": metrics["combined_accuracy"],
                }
            )

        snr_df = pd.DataFrame(snr_data)
        snr_df.to_csv(os.path.join(eval_dir, "results_by_snr.csv"), index=False)

        # Save overall results
        overall_df = pd.DataFrame(
            [
                {
                    "eval_loss": eval_loss / num_batches,
                    "spherical_error": overall_results["spherical_error"],
                    "azimuth_error": overall_results["azimuth_error"],
                    "elevation_error": overall_results["elevation_error"],
                    "azimuth_accuracy": overall_results["azimuth_accuracy"],
                    "elevation_accuracy": overall_results["elevation_accuracy"],
                    "combined_accuracy": overall_results["combined_accuracy"],
                }
            ]
        )
        overall_df.to_csv(os.path.join(eval_dir, "overall_results.csv"), index=False)

        # Save confusion matrices
        for key in ("cm_azim", "cm_elev", "cm_class"):
            np.save(os.path.join(eval_dir, f"{key}.npy"), conf_matrices[key])

        print(f"Results saved to {eval_dir}/")

    def _get_bm_params(self) -> dict | None:
        """Returns the learned BM fc/ourbeta params, if any BM layer is learnable."""
        from auditory_layers.cochlear import BM

        for module in self.model.modules():
            if isinstance(module, BM) and module.learnable_coefficients:
                return module.get_learned_params()
        return None

    def _save_bm_history(self) -> None:
        if not self.bm_fc_history:
            return
        history_dir = self.exp_dirs["history"]
        # shape: (num_epochs, num_filters)
        np.save(
            os.path.join(history_dir, "bm_fc_history.npy"), np.stack(self.bm_fc_history)
        )
        np.save(
            os.path.join(history_dir, "bm_ourbeta_history.npy"),
            np.stack(self.bm_ourbeta_history),
        )

    def _save_checkpoint(self) -> None:
        """Save temporary best model to checkpoint directory"""
        checkpoint_path = os.path.join(self.exp_dirs["checkpoint"], "best_model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)

    def _load_checkpoint(self) -> None:
        """Load best model from checkpoint"""
        checkpoint_path = os.path.join(self.exp_dirs["checkpoint"], "best_model.pth")
        self.model.load_state_dict(torch.load(checkpoint_path))

    def _save_final_model(self) -> None:
        """Save final trained model"""
        model_path = os.path.join(self.exp_dirs["model"], "final_model.pth")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self.config,
            },
            model_path,
        )
        print(f"Final trained model saved to {model_path}")

    def _save_losses(self) -> None:
        """Save training history"""
        loss_path = os.path.join(self.exp_dirs["history"], "training_loss.csv")
        pd.DataFrame(
            {
                "train_loss": self.train_losses,
                "val_loss": self.val_losses,
            }
        ).to_csv(loss_path, index=False)
        print(f"Losses saved to {loss_path}")
