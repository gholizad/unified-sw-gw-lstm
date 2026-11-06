from pathlib import Path
import os
import sys

import torch
import neuralhydrology.datasetzoo as datasetzoo
import neuralhydrology.evaluation.tester as tester
import neuralhydrology.training.basetrainer as basetrainer
from neuralhydrology.nh_run import start_run

try:
    from gwlstm import GWLstm
except ImportError:
    GWLstm = None

try:
    from srsblstm import SrSbLstm
except ImportError:
    SrSbLstm = None


original_get_dataset = datasetzoo.get_dataset


def get_dataset_with_custom_lstm(*args, **kwargs):
    cfg = kwargs.get("cfg", args[0] if args else None)
    if cfg is not None:
        dataset = str(cfg.dataset).lower()
        if dataset == "sr_sb_lstm" and SrSbLstm is not None:
            return SrSbLstm(*args, **kwargs)
        if dataset == "gw_lstm" and GWLstm is not None:
            return GWLstm(*args, **kwargs)
    return original_get_dataset(*args, **kwargs)


def newest_pantarhei_config() -> Path:
    configs = list(Path(".").glob("*_pantarhei.yml"))
    if not configs:
        raise FileNotFoundError("No *_pantarhei.yml files found in the current folder.")
    return max(configs, key=lambda p: p.stat().st_mtime)


datasetzoo.get_dataset = get_dataset_with_custom_lstm
basetrainer.get_dataset = get_dataset_with_custom_lstm
tester.get_dataset = get_dataset_with_custom_lstm

config_file = Path(os.environ["CONFIG"]) if os.environ.get("CONFIG") else Path(sys.argv[1]) if len(sys.argv) > 1 else newest_pantarhei_config()
gpu = int(os.environ.get("GPU", "0"))

print(f"Using config file: {config_file}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device {gpu}: {torch.cuda.get_device_name(gpu)}")

start_run(config_file=config_file, gpu=gpu)
