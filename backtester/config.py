import json
import yaml


def load_config(path: str) -> dict:
    """
    Load a configuration from YAML or JSON.
    """
    with open(path, 'r') as f:
        text = f.read()
        if path.endswith(('.yml', '.yaml')):
            return yaml.safe_load(text)
        elif path.endswith('.json'):
            return json.loads(text)
        else:
            raise ValueError(f"Unsupported config format: {path}")
