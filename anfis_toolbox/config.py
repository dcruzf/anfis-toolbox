"""Configuration utilities for ANFIS models."""

import json
import logging
import pickle  # nosec B403
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, Protocol, TypedDict, cast

from .builders import ANFISBuilder
from .model import TSKANFIS


class _InputConfig(TypedDict):
    range_min: float
    range_max: float
    n_mfs: int
    mf_type: str
    overlap: float


class _TrainingConfigRequired(TypedDict):
    method: str
    epochs: int
    learning_rate: float


class _TrainingConfigOptional(TypedDict, total=False):
    verbose: bool


class _TrainingConfig(_TrainingConfigRequired, _TrainingConfigOptional):
    pass


class _ConfigDict(TypedDict):
    inputs: dict[str, _InputConfig]
    training: _TrainingConfig
    model_params: dict[str, Any]


class _PresetConfig(TypedDict):
    description: str
    inputs: dict[str, _InputConfig]
    training: _TrainingConfig


class _SupportsParameters(Protocol):
    @property
    def parameters(self) -> dict[str, Any]:  # pragma: no cover - protocol definition
        """Return membership function parameters."""
        ...


_MembershipConfig = dict[str, list[dict[str, Any]]]


class ANFISConfig:
    """Configuration manager for ANFIS models."""

    def __init__(self) -> None:
        """Initialize configuration manager."""
        self.config: _ConfigDict = {
            "inputs": {},
            "training": {"method": "hybrid", "epochs": 50, "learning_rate": 0.01, "verbose": False},
            "model_params": {},
        }

    def add_input_config(
        self,
        name: str,
        range_min: float,
        range_max: float,
        n_mfs: int = 3,
        mf_type: str = "gaussian",
        overlap: float = 0.5,
    ) -> "ANFISConfig":
        """Add input configuration.

        Parameters:
            name: Input variable name
            range_min: Minimum input range
            range_max: Maximum input range
            n_mfs: Number of membership functions
            mf_type: Type of membership functions
            overlap: Overlap factor

        Returns:
            Self for method chaining
        """
        self.config["inputs"][name] = {
            "range_min": range_min,
            "range_max": range_max,
            "n_mfs": n_mfs,
            "mf_type": mf_type,
            "overlap": overlap,
        }
        return self

    def set_training_config(
        self, method: str = "hybrid", epochs: int = 50, learning_rate: float = 0.01, verbose: bool = False
    ) -> "ANFISConfig":
        """Set training configuration.

        Parameters:
            method: Training method ('hybrid' or 'backprop')
            epochs: Number of training epochs
            learning_rate: Learning rate
            verbose: Whether to show training progress

        Returns:
            Self for method chaining
        """
        self.config["training"].update(
            {"method": method, "epochs": epochs, "learning_rate": learning_rate, "verbose": verbose}
        )
        return self

    def build_model(self) -> TSKANFIS:
        """Build ANFIS model from configuration.

        Returns:
            Configured ANFIS model
        """
        if not self.config["inputs"]:
            raise ValueError("No inputs configured. Use add_input_config() first.")

        builder = ANFISBuilder()

        inputs = self.config["inputs"]

        for name, params in inputs.items():
            builder.add_input(
                name=name,
                range_min=float(params["range_min"]),
                range_max=float(params["range_max"]),
                n_mfs=int(params["n_mfs"]),
                mf_type=str(params["mf_type"]),
                overlap=float(params["overlap"]),
            )

        return builder.build()

    def save(self, filepath: str | Path) -> None:
        """Save configuration to JSON file.

        Parameters:
            filepath: Path to save configuration file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(self.config, f, indent=2)

    @classmethod
    def load(cls, filepath: str | Path) -> "ANFISConfig":
        """Load configuration from JSON file.

        Parameters:
            filepath: Path to configuration file

        Returns:
            ANFISConfig object
        """
        with open(filepath) as f:
            config_data = json.load(f)

        config = cls()
        config.config = cast(_ConfigDict, config_data)
        return config

    def to_dict(self) -> _ConfigDict:
        """Convert configuration to dictionary.

        Returns:
            Configuration dictionary
        """
        return deepcopy(self.config)

    def __repr__(self) -> str:
        """String representation of configuration."""
        inputs = self.config["inputs"]
        n_inputs = len(inputs)
        total_mfs = sum(int(inp["n_mfs"]) for inp in inputs.values())

        return f"ANFISConfig(inputs={n_inputs}, total_mfs={total_mfs}, method={self.config['training']['method']})"


class ANFISModelManager:
    """Model management utilities for saving/loading trained ANFIS models."""

    @staticmethod
    def save_model(model: TSKANFIS, filepath: str | Path, include_config: bool = True) -> None:
        """Save trained ANFIS model to file.

        Parameters:
            model: Trained ANFIS model
            filepath: Path to save model file
            include_config: Whether to save model configuration
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save model using pickle
        with open(filepath, "wb") as f:
            pickle.dump(model, f)  # nosec B301

        # Save configuration if requested
        if include_config:
            config_path = filepath.with_suffix(".config.json")
            try:
                config = ANFISModelManager._extract_config(model)
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)
            except Exception as e:
                logging.warning("Could not save model configuration: %s", e)

    @staticmethod
    def load_model(filepath: str | Path) -> TSKANFIS:
        """Load trained ANFIS model from file.

        Parameters:
            filepath: Path to model file

        Returns:
            Loaded ANFIS model
        """
        with open(filepath, "rb") as f:
            model: TSKANFIS = pickle.load(f)  # nosec B301

        return model

    @staticmethod
    def _extract_config(model: TSKANFIS) -> dict[str, Any]:
        """Extract configuration from trained model.

        Parameters:
            model: ANFIS model

        Returns:
            Model configuration dictionary
        """
        # Use standardized interface: both model and membership_layer have membership_functions property
        membership_functions: Mapping[str, Sequence[_SupportsParameters]] = model.membership_functions
        input_names = model.input_names

        membership_config: _MembershipConfig = {}
        config: dict[str, Any] = {
            "model_info": {
                "n_inputs": int(model.n_inputs),
                "n_rules": int(model.n_rules),
                "input_names": input_names,
            },
            "membership_functions": membership_config,
        }

        # Extract MF information from each input channel
        for input_name, mfs in membership_functions.items():
            membership_config[input_name] = []

            for _i, mf in enumerate(mfs):
                # Convert numpy scalars to native Python types for JSON serialization
                parameters: dict[str, Any] = mf.parameters.copy()
                for key, value in parameters.items():
                    if hasattr(value, "item"):
                        parameters[key] = value.item()

                mf_info = {"type": mf.__class__.__name__, "parameters": parameters}
                membership_config[input_name].append(mf_info)

        return config


# Predefined configurations for common use cases
PREDEFINED_CONFIGS: dict[str, _PresetConfig] = {
    "1d_function": {
        "description": "Single input function approximation",
        "inputs": {"x": {"range_min": -5, "range_max": 5, "n_mfs": 5, "mf_type": "gaussian", "overlap": 0.5}},
        "training": {"method": "hybrid", "epochs": 100, "learning_rate": 0.01},
    },
    "2d_regression": {
        "description": "Two-input regression problem",
        "inputs": {
            "x1": {"range_min": -2, "range_max": 2, "n_mfs": 3, "mf_type": "gaussian", "overlap": 0.5},
            "x2": {"range_min": -2, "range_max": 2, "n_mfs": 3, "mf_type": "gaussian", "overlap": 0.5},
        },
        "training": {"method": "hybrid", "epochs": 50, "learning_rate": 0.01},
    },
    "control_system": {
        "description": "Control system with error and error rate",
        "inputs": {
            "error": {"range_min": -1, "range_max": 1, "n_mfs": 5, "mf_type": "triangular", "overlap": 0.3},
            "error_rate": {"range_min": -1, "range_max": 1, "n_mfs": 5, "mf_type": "triangular", "overlap": 0.3},
        },
        "training": {"method": "hybrid", "epochs": 75, "learning_rate": 0.015},
    },
    "time_series": {
        "description": "Time series prediction with lag inputs",
        "inputs": {
            "lag1": {"range_min": -3, "range_max": 3, "n_mfs": 4, "mf_type": "gaussian", "overlap": 0.4},
            "lag2": {"range_min": -3, "range_max": 3, "n_mfs": 4, "mf_type": "gaussian", "overlap": 0.4},
            "lag3": {"range_min": -3, "range_max": 3, "n_mfs": 3, "mf_type": "gaussian", "overlap": 0.4},
        },
        "training": {"method": "hybrid", "epochs": 60, "learning_rate": 0.008},
    },
}


def create_config_from_preset(preset_name: str) -> ANFISConfig:
    """Create configuration from predefined preset.

    Parameters:
        preset_name: Name of predefined configuration

    Returns:
        ANFISConfig object

    Raises:
        ValueError: If preset name not found
    """
    if preset_name not in PREDEFINED_CONFIGS:
        available = list(PREDEFINED_CONFIGS.keys())
        raise ValueError(f"Preset '{preset_name}' not found. Available presets: {available}")

    preset = PREDEFINED_CONFIGS[preset_name]
    config = ANFISConfig()

    # Add inputs
    for name, params in preset["inputs"].items():
        config.add_input_config(name, **params)

    # Set training parameters
    config.set_training_config(**preset["training"])

    return config


def list_presets() -> dict[str, str]:
    """List available predefined configurations.

    Returns:
        Dictionary mapping preset names to descriptions
    """
    return {name: info["description"] for name, info in PREDEFINED_CONFIGS.items()}
