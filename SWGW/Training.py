from pathlib import Path
import os
import sys

import torch
import neuralhydrology.datasetzoo as datasetzoo
import neuralhydrology.evaluation.tester as tester
import neuralhydrology.training.basetrainer as basetrainer
from neuralhydrology.nh_run import start_run

from srsblstm import SrSbLstm
from nse_training_runtime import install


original_get_dataset = datasetzoo.get_dataset


def get_dataset_with_custom_lstm(*args, **kwargs):
    cfg = kwargs.get("cfg", args[0] if args else None)
    if cfg is not None and str(cfg.dataset).lower() == "sr_sb_lstm":
        return SrSbLstm(*args, **kwargs)
    return original_get_dataset(*args, **kwargs)


datasetzoo.get_dataset = get_dataset_with_custom_lstm
basetrainer.get_dataset = get_dataset_with_custom_lstm
tester.get_dataset = get_dataset_with_custom_lstm


def main():
    config_file = (
        Path(os.environ["CONFIG"]) if os.environ.get("CONFIG")
        else Path(sys.argv[1]) if len(sys.argv) > 1
        else Path(__file__).with_name("config.yml")
    )
    config_file = config_file.resolve()
    os.chdir(config_file.parent)
    gpu = int(os.environ.get("GPU", "0"))
    install(config_file)
    start_run(config_file=config_file, gpu=gpu)


if __name__ == "__main__":
    main()
