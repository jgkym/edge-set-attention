from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class EmbeddingConfig(BaseModel):
    """Configuration for the embedding layer."""

    node_dim: int = 61
    edge_dim: int = 13
    mlp_hidden_dim: int = 128
    embedding_dim: int = 128


class ESAConfig(BaseModel):
    """Configuration for the Edge-Set Attention block."""

    num_heads: int = 4
    mlp_hidden_dim: int = 128
    num_seeds: int = 32
    blocks: Literal["M", "P", "S"] = (
        "MMMSPS"  # Using str for flexibility, can be validated with pydantic
    )


class ModelConfig(BaseModel):
    """Configuration for the model."""

    dropout: float = 0.4
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    esa: ESAConfig = Field(default_factory=ESAConfig)


class DataConfig(BaseModel):
    """Configuration for data loading."""

    data_dir: Path = Path("data")
    filename: str = "train_cleaned.csv"
    smiles_col: str = "smiles"
    target_col: str | None = "pIC50"
    one_hot: bool = True
    max_atomic_number: int = 53
    pe_types: list[str] = Field(
        default_factory=lambda: ["LapPE", "RWSE", "EquivStableLapPE"]
    )  # "LapPE", "RWSE", "EquivStableLapPE"
    train_ratio: float | None = 0.8


class TrainingConfig(BaseModel):
    """Configuration for training."""

    batch_size: int = 32
    epochs: int = 200

    # --- Optimizer ---
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    gradient_accumulation_steps: int = 2
    early_stopping_patience: int = 30

    # --- Logging ---
    report_to: str = "trackio"
    project_name: str = "ESA"
    logging_steps: int = 10

    # --- Saving ---
    output_dir: Path = Path("outputs")
    metric_for_best_model: str = "loss"
    greater_is_better: bool = False


class AppConfig(BaseModel):
    """Base configuration for the project."""

    random_seed: int = 7
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)


def get_config() -> AppConfig:
    """Initializes and returns the application configuration."""
    return AppConfig()
