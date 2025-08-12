from pathlib import Path

import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import TorchDynamoPlugin
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from esa.metrics import compute_metrics


class Trainer:
    def __init__(
        self,
        config,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
    ) -> None:
        self.config = config

        # --- Accelerator ---
        # Initialize accelerator first. It will handle device placement, model wrapping, etc.
        # Configure TorchDynamoPlugin for torch.compile if available
        # torch_dynamo_plugin = (
        #     TorchDynamoPlugin(
        #         backend="inductor",
        #         mode="max-autotune",
        #         use_regional_compilation=True,
        #     )
        #     if hasattr(torch, "compile")
        #     else None
        # )

        self.accelerator = Accelerator(
            log_with=config.report_to,
            # mixed_precision="fp16",  # Use "bf16" on A100/H100, "fp16" on others
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            # dynamo_plugin=torch_dynamo_plugin,
        )
        self.device = self.accelerator.device

        print("Starting Training...")
        # Let accelerate handle the device printing
        self.accelerator.print(f"Using device: {self.accelerator.device}")

        # --- Optimizer, Scheduler, and Loss ---
        optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
        )
        self.criterion = nn.HuberLoss()

        # --- Prepare everything with Accelerator ---
        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.val_loader,
            self.scheduler,
        ) = self.accelerator.prepare(
            model, optimizer, train_loader, val_loader, scheduler
        )

        # --- State Tracking & Early Stopping ---
        self.best_loss = float("inf")
        self.early_stopping_patience = config.early_stopping_patience
        self.patience_counter = 0

        # --- Output & Logging ---
        if self.accelerator.is_main_process and config.report_to is not None:
            self.accelerator.init_trackers(project_name="ESA", config=vars(config))

        self.output_dir = Path(config.output_dir)
        # Let the main process create the directory
        if self.accelerator.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_path = self.output_dir / "best_model_state"

        # Check if torch.compile is being used, for conditional optimizations
        self.use_torch_compile = (
            self.accelerator.state.dynamo_plugin is not None
            and hasattr(torch, "compile")
        )

    def train(self) -> Path:
        """
        Main training loop. Handles training, validation, model saving, and early stopping.

        Returns:
            Path: The path to the best saved model checkpoint directory.
        """
        for epoch in range(self.config.epochs):
            train_metrics = self._train_epoch(epoch)
            train_loss = train_metrics["loss"]
            log_metrics = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_score": train_metrics["score"],
                "train_nrmse": train_metrics["nrmse"],
                "train_r2": train_metrics["r2"],
            }

            log_message = f"Epoch {epoch + 1}/{self.config.epochs}\nTrain Loss: {train_loss:.4f} | Train Score: {train_metrics['score']:.4f} | Train NRMSE: {train_metrics['nrmse']:.4f} | Train R2: {train_metrics['r2']:.4f}"

            # Determine loss for early stopping and logging
            if self.val_loader:
                eval_metrics = self._evaluate()
                current_loss = eval_metrics["loss"]
                log_metrics.update(eval_metrics)
                log_message += f"\nVal Loss: {eval_metrics['loss']:.4f} | Val Score: {eval_metrics['score']:.4f} | Val NRMSE: {eval_metrics['nrmse']:.4f} | Val R2: {eval_metrics['r2']:.4f}"
            else:
                current_loss = train_loss

            # Scheduler should be stepped once per epoch
            self.scheduler.step()

            if current_loss < self.best_loss:
                self.patience_counter = 0
                self.best_loss = current_loss
                log_message += f"\nNew best loss: {self.best_loss:.4f}. Saving model..."
                # `save_state` handles unwrapping the model and saving from the main process only.
                self.accelerator.save_state(self.best_model_path)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    log_message += "\nEarly stopping triggered."
                    self.accelerator.print(log_message)
                    break

            if self.config.report_to is not None:
                self.accelerator.log(log_metrics, step=epoch)
            self.accelerator.print(log_message)

        self.accelerator.print(
            f"\n\nTraining completed. Best validation loss: {self.best_loss:.4f}"
        )
        self.accelerator.print(f"Best model state saved at: {self.best_model_path}")

        if self.config.report_to is not None:
            self.accelerator.end_training()

        return self.best_model_path

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        """Runs a single training epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.config.epochs}",
            disable=not self.accelerator.is_main_process,
        )

        for i, inputs in enumerate(progress_bar):
            targets = inputs.y
            if self.use_torch_compile:
                # This tells the compiler that a new iteration is starting,
                # preventing it from overwriting memory needed by the previous step's backward pass.
                torch.compiler.cudagraph_mark_step_begin()

            # Data is automatically moved to the correct device by the prepared DataLoader
            with self.accelerator.accumulate(self.model):
                logits = self.model(inputs)
                loss = self.criterion(logits.squeeze(), targets)

                self.accelerator.backward(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Gather loss across all processes for accurate logging
            avg_loss = self.accelerator.gather(loss.detach()).mean()
            total_loss += avg_loss.item()

            all_preds.append(logits.detach())
            all_targets.append(targets.detach())

            if self.accelerator.is_main_process:
                progress_bar.set_postfix(
                    loss=f"{avg_loss.item():.4f}",
                    lr=f"{self.scheduler.get_last_lr()[0]:.1e}",
                )

            if self.config.report_to is not None and i % self.config.logging_steps == 0:
                self.accelerator.log({"train/step_loss": avg_loss.item()})

        # Gather predictions and targets from all processes
        gathered_preds = self.accelerator.gather_for_metrics(torch.cat(all_preds))
        gathered_targets = self.accelerator.gather_for_metrics(torch.cat(all_targets))

        # Compute metrics on the gathered tensors
        metrics = compute_metrics(gathered_preds, gathered_targets)
        metrics["loss"] = total_loss / len(self.train_loader)
        return {k: v.item() if hasattr(v, "item") else v for k, v in metrics.items()}

    def _evaluate(self) -> dict[str, float]:
        """
        Runs a single evaluation epoch, gathering metrics from all processes for accuracy.
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        progress_bar = tqdm(
            self.val_loader,
            desc="Evaluating",
            disable=not self.accelerator.is_main_process,
        )

        with torch.no_grad():
            for inputs in progress_bar:
                targets = inputs.y
                logits = self.model(inputs)
                loss = self.criterion(logits.squeeze(), targets)

                avg_loss = self.accelerator.gather(loss.detach()).mean()
                total_loss += avg_loss.item()

                all_preds.append(logits.detach())
                all_targets.append(targets.detach())

                if self.accelerator.is_main_process:
                    progress_bar.set_postfix(loss=f"{avg_loss.item():.4f}")

        # Gather predictions and targets from all processes
        gathered_preds = self.accelerator.gather_for_metrics(torch.cat(all_preds))
        gathered_targets = self.accelerator.gather_for_metrics(torch.cat(all_targets))

        # Compute metrics on the gathered tensors
        metrics = compute_metrics(gathered_preds, gathered_targets)
        metrics["loss"] = total_loss / len(self.val_loader)
        return {k: v.item() if hasattr(v, "item") else v for k, v in metrics.items()}
