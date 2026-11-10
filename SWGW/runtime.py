"""Training-only normalized NSE statistics for the joint SW-GW model."""
from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import textwrap
import uuid

import numpy as np
import pandas as pd
import torch

TARGETS = ["QObs(mm/d)", "GWD(m)"]
TRAINING_PERIOD = ["2010-01-01", "2021-12-31"]
EPSILON = 0.1
_installed_context = None


def target_spec(dataset):
    targets = list(dataset.cfg.target_variables)
    require(targets == TARGETS, "Expected joint streamflow and groundwater targets.")
    require(dataset.cfg.target_loss_weights is None, "Use the framework's default loss weighting.")
    return targets, [1 / len(targets) for _ in targets]


def verify_environment():
    from neuralhydrology.datasetzoo.basedataset import BaseDataset
    from neuralhydrology.training.loss import MaskedNSELoss
    from neuralhydrology.utils.config import Config

    require(importlib.metadata.version("neuralhydrology") == "1.13.0",
            "This training helper requires NeuralHydrology 1.13.0.")
    tree = ast.parse(textwrap.dedent(inspect.getsource(BaseDataset._load_data)))
    calls, normalization_lines = {}, []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            calls.setdefault(node.func.attr, []).append(node.lineno)
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "xr" for target in node.targets
        ):
            if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Div):
                code = ast.unparse(node.value)
                if "xarray_feature_center" in code and "xarray_feature_scale" in code:
                    normalization_lines.append(node.lineno)
    for method in ("_calculate_per_basin_std", "_setup_normalization", "_create_lookup_table"):
        require(len(calls.get(method, [])) == 1, f"Unsupported dataset lifecycle: {method}")
    require(
        len(normalization_lines) == 1
        and calls["_calculate_per_basin_std"][0] < calls["_setup_normalization"][0]
        < normalization_lines[0] < calls["_create_lookup_table"][0],
        "Normalized targets must reach the lookup-table hook.",
    )
    cfg = Config({"target_variables": TARGETS, "predict_last_n": 1, "loss": "NSE"})
    loss = MaskedNSELoss(cfg)
    require(loss.eps == EPSILON, "Unexpected NSE stabilization constant.")
    observed = torch.tensor([[[1., 2.]], [[3., float("nan")]]])
    stds = torch.tensor([[[2., 1.]], [[4., 1.]]])
    predicted = torch.zeros_like(observed, requires_grad=True)
    actual, _ = loss({"y_hat": predicted}, {"y": observed, "per_basin_target_stds": stds})
    expected = 0.0
    for index in range(len(TARGETS)):
        values = observed[:, :, index].numpy().astype(float)
        denominator = stds[:, :, index].numpy().astype(float)
        valid = np.isfinite(values)
        expected += np.mean(values[valid] ** 2 / (denominator[valid] + EPSILON) ** 2) / len(TARGETS)
    require(np.isclose(actual.item(), expected, rtol=2e-6), "Unexpected masked NSE arithmetic.")
    actual.backward()
    require(torch.isfinite(predicted.grad).all().item(), "Invalid NSE gradients.")
    return {
        "neuralhydrology_version": importlib.metadata.version("neuralhydrology"),
        "torch_version": str(torch.__version__),
        "epsilon": EPSILON,
        "loss_check_passed": True,
    }

def require(condition, message):
    if not condition:
        raise ValueError(message)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    """Write deterministic provenance once, rejecting incompatible later writes."""
    path = Path(path)
    text = json.dumps(value, indent=2, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_text(encoding='utf-8') == text,
                f'Refusing to overwrite changed NSE runtime provenance: {path}')
        return
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)

def target_scalers(dataset):
    targets, _ = target_spec(dataset)
    result = {}
    for target in targets:
        center = float(dataset.scaler['xarray_feature_center'][target].item())
        scale = float(dataset.scaler['xarray_feature_scale'][target].item())
        require(np.isfinite(center) and np.isfinite(scale) and scale > 0,
                f'Invalid framework target normalization: {target}')
        result[target] = {'center': center, 'scale': scale}
    if 'GWD(m)' in targets:
        require(abs(result['GWD(m)']['center']) < 1e-6 and
                abs(result['GWD(m)']['scale'] - 1.0) < 1e-6,
                'Expected frozen per-well GWD inputs with framework center=0 and scale=1')
    return result

def training_statistics(dataset, xr):
    """Calculate each configured target's std from normalized training data only."""
    targets, weights = target_spec(dataset)
    require(dataset.period == 'train' and dataset.is_train,
            'NSE denominators may only be fitted from a training dataset')
    start, end = pd.Timestamp(dataset.cfg.train_start_date), pd.Timestamp(dataset.cfg.train_end_date)
    require([str(start.date()), str(end.date())] == TRAINING_PERIOD,
            'NSE training period differs from the frozen experiment')
    require(dataset.cfg.per_basin_train_periods_file is None,
            'Per-basin period overrides are not supported in this experiment')
    dates = pd.DatetimeIndex(xr['date'].values)
    require(not dates.has_duplicates and dates.is_monotonic_increasing,
            'Duplicated or unordered normalized training dates')
    selected = xr.sel(date=slice(start, end))
    require(selected.sizes.get('date', 0) > 1, 'No NSE training observations')
    basins = list(dataset.basins)
    require(len(basins) == len(set(basins)), 'Duplicate training basins')
    records = {}
    for basin in basins:
        records[basin] = {}
        for target in targets:
            data = selected.sel(basin=basin)[target]
            require(tuple(data.dims) == ('date',), 'Expected one daily target series per basin')
            values = np.asarray(data.values, dtype=np.float64)
            require(not np.isinf(values).any(), f'Infinite {target} training observation: {basin}')
            finite = values[np.isfinite(values)]
            require(len(finite) >= 2, f'Fewer than two {target} training values: {basin}')
            std = float(np.std(finite, ddof=0))
            require(np.isfinite(std) and std >= 0, f'Invalid training std: {basin} {target}')
            records[basin][target] = {'std_normalized': std, 'n_training_observations': len(finite)}
    return {'schema_version': 2, 'training_period': TRAINING_PERIOD,
            'targets': targets, 'target_loss_weights': weights, 'basins': basins, 'std_ddof': 0,
            'framework_target_scalers': target_scalers(dataset), 'basin_statistics': records,
            'epsilon_normalized': EPSILON,
            'formula': 'sum_Y weight_Y*mean(valid Y squared errors/(training basin Y std+0.1)^2)',
            'space': 'Normalized target space, exactly the space of model errors',
            'fitting_data': 'Training target observations only; no validation or test observations',
            'aggregation': 'Mean over valid samples separately for each target; not exact macro basin mean NSE',
            'zero_std_policy': 'Allowed; denominator is epsilon squared (stabilized surrogate)'}

def attach_statistics(dataset, record):
    targets, weights = target_spec(dataset)
    require(record.get('targets') == targets and record.get('training_period') == TRAINING_PERIOD,
            'NSE denominator provenance has incompatible targets or training dates')
    require(record.get('target_loss_weights') == weights and record.get('std_ddof') == 0,
            'NSE denominator provenance has incompatible task weights or std convention')
    require(record.get('epsilon_normalized') == EPSILON, 'NSE epsilon changed')
    current = target_scalers(dataset)
    for target in targets:
        for key in ('center', 'scale'):
            require(np.isclose(current[target][key], record['framework_target_scalers'][target][key],
                               rtol=1e-6, atol=1e-7),
                    f'Evaluation uses a different framework scaler: {target} {key}')
    result = {}
    for basin in dataset.basins:
        require(basin in record['basin_statistics'], f'No training NSE denominator for {basin}')
        values = [record['basin_statistics'][basin][target]['std_normalized'] for target in targets]
        require(np.isfinite(values).all() and min(values) >= 0, f'Invalid saved training std: {basin}')
        result[basin] = torch.tensor([values], dtype=torch.float32)
    dataset._per_basin_target_stds = result


def install(config_file):
    """Install the training-statistics hooks for this configured run."""
    global _installed_context
    from neuralhydrology.utils.config import Config
    from neuralhydrology.datasetzoo.basedataset import BaseDataset

    config = Config(Path(config_file).resolve(strict=True))
    require(config.model == "cudalstm" and config.dataset == "sr_sb_lstm",
            "Expected the joint LSTM model and dataset.")
    require(config.loss.lower() == "nse", "This helper requires loss: NSE.")
    require(list(config.target_variables) == TARGETS, "Unexpected prediction targets.")
    require(config.target_loss_weights is None, "Omit fixed loss weights from the configuration.")
    require(config.run_dir is not None, "Configure a relative output directory.")
    run_parent = Path(config.run_dir).resolve()
    if _installed_context is not None:
        require(_installed_context == run_parent, "NSE hooks are already installed for another run root.")
        return
    environment = verify_environment()
    original_stds = BaseDataset._calculate_per_basin_std
    original_lookup = BaseDataset._create_lookup_table

    def in_scope(dataset):
        run = Path(dataset.cfg.run_dir).resolve()
        return run.parent == run_parent

    def guard(dataset):
        require(dataset.cfg.loss.lower() == "nse", "Expected NSE training.")
        target_spec(dataset)
        require(dataset.cfg.predict_last_n == 1 and not dataset.cfg.use_frequencies,
                "Use one daily prediction at a single frequency.")
        require(dataset.cfg.target_noise_std is None, "Target noise must be disabled.")

    def defer_pre_normalization_stds(dataset, xr):
        if not in_scope(dataset):
            return original_stds(dataset, xr)
        guard(dataset)
        dataset._per_basin_target_stds = {}

    def normalized_lookup(dataset, xr):
        if not in_scope(dataset):
            return original_lookup(dataset, xr)
        guard(dataset)
        run = Path(dataset.cfg.run_dir).resolve()
        path = run / "train_data" / "nse_training_stds.json"
        checksum = path.with_suffix(".json.sha256")
        if dataset.is_train:
            record = training_statistics(dataset, xr)
            write_json(path, record)
            expected = digest(path)
            if checksum.exists():
                require(checksum.read_text().strip() == expected, "Training statistics checksum changed.")
            else:
                checksum.write_text(expected + "\n", encoding="ascii")
        else:
            require(path.is_file() and checksum.is_file(), "Training NSE statistics are missing.")
            require(digest(path) == checksum.read_text().strip(), "Training NSE statistics changed.")
            record = json.loads(path.read_text(encoding="utf-8"))
        attach_statistics(dataset, record)
        write_json(run / "NSE_RUNTIME.json", {
            "model": "SWGW",
            "environment": environment,
            "targets": TARGETS,
            "training_period": TRAINING_PERIOD,
            "method": "Training-only basin deviations calculated after target normalization",
            "nh_installation_modified": False,
        })
        return original_lookup(dataset, xr)

    BaseDataset._calculate_per_basin_std = defer_pre_normalization_stds
    BaseDataset._create_lookup_table = normalized_lookup
    _installed_context = run_parent
