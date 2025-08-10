import torch.nn as nn
from pathlib import Path
from typing import Literal, Type
from pydantic import BaseModel, Field

# A map to resolve activation function from a string representation
ACTIVATION_FUNCTIONS = {
    "ReLU": nn.ReLU,
    "SiLU": nn.SiLU,
    "GELU": nn.GELU,
}

class EmbeddingConfig(BaseModel):
    """Configuration for the embedding layer."""
    node_dim: int = 61
    edge_dim: int = 13
    global_dim: int = 6
    hidden_dims: list[int] = Field(default_factory=lambda: [512, 512])

class ESAConfig(BaseModel):
    """Configuration for the Edge-Set Attention block."""
    num_heads: int = 4
    mlp_hidden_dims: list[int] = Field(default_factory=lambda: [512, 256, 512])
    num_seeds: int = 32
    blocks: Literal["M", "P", "S"] = "SMMMMPS" # Using str for flexibility, can be validated with pydantic

class ModelConfig(BaseModel):
    """Configuration for the model."""
    activation_fn_name: str = "ReLU"
    dropout: float | None = 0.1
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    esa: ESAConfig = Field(default_factory=ESAConfig)

    def get_activation_fn(self) -> Type[nn.Module]:
        """Returns the activation function class from its name."""
        try:
            return ACTIVATION_FUNCTIONS[self.activation_fn_name]
        except KeyError:
            raise ValueError(f"Unsupported activation function: {self.activation_fn_name}")

class DataConfig(BaseModel):
    """Configuration for data loading."""
    data_dir: Path = Path("data")
    filename: str = "train_cleaned.csv"
    smiles_col: str = "smiles"
    target_col: str | None = "pIC50"
    one_hot: bool = True
    max_atomic_number: int = 35
    pe_types: list[str] = Field(default_factory=lambda: ["LapPE", "RWSE", "EquivStableLapPE"]) # "LapPE", "RWSE", "EquivStableLapPE" 
    train_ratio: float | None = 0.8

class TrainingConfig(BaseModel):
    """Configuration for training."""
    batch_size: int = 32
    epochs: int = 50
    
    # --- Optimizer --- 
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    gradient_accumulation_steps: int = 2
    early_stopping_patience: int = 50
    
    # --- Logging ---
    report_to: str = "trackio"
    logging_steps: int = 10
    
    # --- Saving ---
    output_dir: Path = Path("outputs")

class AppConfig(BaseModel):
    """Base configuration for the project."""
    random_seed: int = 0
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

def get_config() -> AppConfig:
    """Initializes and returns the application configuration."""
    return AppConfig()