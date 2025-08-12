from pathlib import Path

import torch
import torch.nn as nn
from accelerate import Accelerator
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

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
        self.accelerator = Accelerator(
            log_with=config.report_to,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
        )
        self.device = self.accelerator.device

        # --- Rich Console for Pretty Logging (Main Process Only) ---
        self.console = Console() if self.accelerator.is_main_process else None

        if self.console:
            self.console.print("[bold green]Starting Training...[/bold green]")
            self.console.print(f"Using device: {self.accelerator.device}")

        # --- Optimizer, Scheduler, and Loss ---
        optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
        self.criterion = nn.MSELoss()

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
        self.best_metric = float("-inf") if config.greater_is_better else float("inf")
        self.early_stopping_patience = config.early_stopping_patience
        self.patience_counter = 0

        # --- Output & Logging ---
        if self.accelerator.is_main_process and config.report_to is not None:
            self.accelerator.init_trackers(
                project_name=config.project_name, config=vars(config)
            )

        self.output_dir = Path(config.output_dir)
        if self.accelerator.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_path = self.output_dir / "best_model_state"

    def train(self) -> Path:
        """
        Main training loop. Handles training, validation, model saving, and early stopping.
        """
        progress = None
        if self.console:
            progress_columns = [
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                TimeElapsedColumn(),
            ]
            progress = Progress(*progress_columns, console=self.console)

        try:
            if progress:
                progress.start()

            epochs_task = None
            if progress:
                epochs_task = progress.add_task(
                    "[cyan]Epochs", total=self.config.epochs
                )

            for epoch in range(self.config.epochs):
                train_metrics = self._train_epoch(epoch, progress)

                eval_metrics = {}
                if self.val_loader:
                    eval_metrics.update(self._evaluate("val/", progress))

                self.scheduler.step()
                log_metrics = {**train_metrics, **eval_metrics, "epoch": epoch + 1}

                if self._log_and_check_early_stopping(log_metrics):
                    break

                if progress:
                    progress.update(epochs_task, advance=1)
        finally:
            if progress:
                progress.stop()

        if self.console:
            best_split = "validation" if self.val_loader else "training"
            self.console.print(
                Panel(
                    f"Best {best_split} {self.config.metric_for_best_model}: {self.best_metric:.4f}\n"
                    f"Best model state saved at: {self.best_model_path}",
                    title="[bold blue]Training Completed[/bold blue]",
                    expand=False,
                )
            )

        if self.config.report_to is not None:
            self.accelerator.end_training()

        return self.best_model_path

    def _train_epoch(self, epoch: int, progress: Progress | None) -> dict[str, float]:
        """Runs a single training epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds, all_targets = [], []

        task_id = None
        if progress:
            task_id = progress.add_task(
                f"[blue]Train Epoch {epoch + 1}", total=len(self.train_loader)
            )

        for inputs in self.train_loader:
            targets = inputs.y
            with self.accelerator.accumulate(self.model):
                logits = self.model(inputs)
                loss = self.criterion(logits.squeeze(), targets)
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()

            avg_loss = self.accelerator.gather(loss.detach()).mean()
            total_loss += avg_loss.item()
            all_preds.append(logits.detach())
            all_targets.append(targets.detach())

            if progress:
                progress.update(
                    task_id,
                    advance=1,
                    description=f"[blue]Train Epoch {epoch + 1} | Loss: {avg_loss.item():.4f}",
                )

        if progress:
            progress.remove_task(task_id)

        gathered_preds = self.accelerator.gather_for_metrics(torch.cat(all_preds))
        gathered_targets = self.accelerator.gather_for_metrics(torch.cat(all_targets))

        metrics = compute_metrics(gathered_preds, gathered_targets, prefix="train/")
        metrics["train/loss"] = total_loss / len(self.train_loader)
        return {k: v.item() if hasattr(v, "item") else v for k, v in metrics.items()}

    def _evaluate(
        self, prefix: str = "val/", progress: Progress | None = None
    ) -> dict[str, float]:
        """Runs a single evaluation epoch."""
        self.model.eval()
        loader = self.val_loader
        if not loader:
            return {}

        total_loss = 0.0
        all_preds, all_targets = [], []

        task_id = None
        if progress:
            task_id = progress.add_task("[purple]Evaluating", total=len(loader))

        with torch.no_grad():
            for inputs in loader:
                targets = inputs.y
                logits = self.model(inputs)
                loss = self.criterion(logits.squeeze(), targets)

                avg_loss = self.accelerator.gather(loss.detach()).mean()
                total_loss += avg_loss.item()
                all_preds.append(logits.detach())
                all_targets.append(targets.detach())

                if progress:
                    progress.update(
                        task_id,
                        advance=1,
                        description=f"[purple]Evaluating | Loss: {avg_loss.item():.4f}",
                    )

        if progress:
            progress.remove_task(task_id)

        gathered_preds = self.accelerator.gather_for_metrics(torch.cat(all_preds))
        gathered_targets = self.accelerator.gather_for_metrics(torch.cat(all_targets))

        metrics = compute_metrics(gathered_preds, gathered_targets, prefix=prefix)
        metrics[f"{prefix}loss"] = total_loss / len(loader)
        return {k: v.item() if hasattr(v, "item") else v for k, v in metrics.items()}

    def _log_and_check_early_stopping(self, metrics: dict) -> bool:
        """Logs metrics using rich and checks for early stopping criteria."""
        monitor_metric_key = f"{'val' if self.val_loader else 'train'}/{self.config.metric_for_best_model}"
        current_metric = metrics[monitor_metric_key]

        improved = (
            self.config.greater_is_better and current_metric > self.best_metric
        ) or (not self.config.greater_is_better and current_metric < self.best_metric)

        if self.console:
            table = Table(title=f"Epoch {metrics['epoch']}/{self.config.epochs}")
            table.add_column("Split", style="cyan")
            table.add_column("Loss", style="magenta")
            table.add_column("Score", style="green")
            table.add_column("NRMSE", style="yellow")
            table.add_column("R2", style="red")

            for split in ["train", "val", "test"]:
                if f"{split}/loss" in metrics:
                    table.add_row(
                        split.capitalize(),
                        f"{metrics[f'{split}/loss']:.4f}",
                        f"{metrics[f'{split}/score']:.4f}",
                        f"{metrics[f'{split}/nrmse']:.4f}",
                        f"{metrics[f'{split}/r2']:.4f}",
                    )
            self.console.print(table)

        if improved:
            self.patience_counter = 0
            self.best_metric = current_metric
            self.accelerator.save_state(self.best_model_path)
            if self.console:
                self.console.print(
                    Panel(
                        f"New best {self.config.metric_for_best_model}: {self.best_metric:.4f}. Saving model.",
                        title="[bold green]Improvement[/bold green]",
                        expand=False,
                    )
                )
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.early_stopping_patience:
                if self.console:
                    self.console.print(
                        Panel(
                            "[bold red]Early stopping triggered.[/bold red]",
                            expand=False,
                        )
                    )
                return True

        if self.config.report_to is not None:
            self.accelerator.log(metrics, step=metrics["epoch"])

        return False
