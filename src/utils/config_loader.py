"""
Responsible for loading and merging YAML configuration files from config/.

Provides a single get_config() function that returns a validated, merged config
dictionary available to all pipeline stages and API components.
"""

import yaml
from pathlib import Path
from typing import Any

# Resolve project root relative to this file's location (src/utils/config_loader.py → ../../)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_config(name: str = "pipeline") -> dict[str, Any]:
    """Load and parse a YAML configuration file from the config/ directory.

    Args:
        name: Config file name without the .yaml extension. Available configs:
            'pipeline', 'models', 'optimization', 'api'.

    Returns:
        Parsed configuration as a nested dictionary.

    Raises:
        FileNotFoundError: If the requested config file does not exist.
        yaml.YAMLError: If the file contains invalid YAML syntax.

    Example:
        >>> cfg = get_config("pipeline")
        >>> cfg["data"]["frequency"]
        'monthly'
    """
    path = _PROJECT_ROOT / "config" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. "
            f"Available configs are in {_PROJECT_ROOT / 'config'}/"
        )
    with open(path) as f:
        return yaml.safe_load(f)
