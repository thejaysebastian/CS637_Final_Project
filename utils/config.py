# usage: config = load_config("configs/base.yaml", "configs/resnet50.yaml")

import yaml

def load_config(base_path, override_path):
    with open(base_path, "r") as f:
        base_config = yaml.safe_load(f)

    with open(override_path, "r") as f:
        override_config = yaml.safe_load(f)

    # simple merge (override takes precedence)
    base_config.update(override_config)

    return base_config