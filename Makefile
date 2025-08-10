train:
	rm -rf data/processed
	uv run python3 -m train

infer:
	rm -rf data/processed
	uv run python3 -m inference

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

.PHONY: clean clean-pyc clean-build clean-torch-cache

# Default target
all:

# Clean up all caches and temporary files
clean: clean-pyc clean-build clean-torch-cache clean-logs clean-jupyter clean-pytest clean-ruff-cache clean-trackio

# Clean Python bytecode files
clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} +

# Clean build artifacts
clean-build:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .coverage
	rm -rf .tox/

# Clean PyTorch specific caches
clean-torch-cache:
	@echo "Cleaning PyTorch cache directories..."
	# PyTorch JIT/TorchScript cache
	rm -rf ~/.cache/torch/jit
	# PyTorch Hub cache (for downloaded models from torchvision, etc.)
	rm -rf ~/.cache/torch/hub
	# Common dataset cache directories (e.g., for torchvision datasets)
	# Note: This might be project-specific if you override default paths.
	# Add any other project-specific dataset cache paths here if known.
	rm -rf ~/.cache/torch/datasets
	rm -rf ~/.cache/torch/models # Older versions might use this
	# Clear the PyTorch custom C++ extensions cache (if any)
	rm -rf $(shell python -c "import torch; print(torch.hub.get_dir())")/extensions
	rm -rf /tmp/torch_extensions # Common temporary location for extensions

# Clean common log files
clean-logs:
	find . -name '*.log' -exec rm -f {} +

# Clean Jupyter notebook checkpoints
clean-jupyter:
	find . -name '.ipynb_checkpoints' -exec rm -rf {} +

# Clean pytest cache
clean-pytest:
	rm -rf .pytest_cache
	rm -rf htmlcov/ # Coverage report directory

# Optional: Clean pip cache (system-wide, not project-specific)
# Use with caution, as it clears cache for ALL pip installations.
# To enable this, add 'clean-pip-cache' to the 'clean' target.
clean-pip-cache:
	@echo "Cleaning pip cache (system-wide)..."
	pip cache purge

clean-ruff-cache:
	rm -rf .ruff_cache

clean-trackio:
	rm -rf ~/.cache/huggingface/trackio