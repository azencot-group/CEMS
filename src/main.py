from __future__ import annotations

import argparse
import logging
import pickle
import time
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
import torch


from scalers import MinMaxScaler
from config import dataset_defaults
from data_generate import load_data
from utils import (
    set_seed,
    get_unique_file_name,
    write_model,
    load_model,
)
from cems_utils import get_probabilities, get_id, shift_bit_length
import algorithm

# --------------------------------------------------------------------------------------
# Logging configuration
# --------------------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# --------------------------------------------------------------------------------------
# Argument handling & reproducibility
# --------------------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """Parse command‑line arguments.

    Returns
    -------
    argparse.Namespace
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="CEMS training / evaluation entry‑point")

    # General I/O ----------------------------------------------------------------------
    parser.add_argument("--result_root_path", type=Path, default=Path("../result"),
                        help="Root directory for experiment outputs.")
    parser.add_argument("--dataset", type=str, default="NO2", help="Dataset key.")
    parser.add_argument("--data_dir", type=Path, default=None,
                        help="Root folder for raw data (RCF_MNIST / TimeSeries).")
    parser.add_argument("--ts_name", type=str, default="", help="Time‑series dataset name.")

    # Reproducibility & compute ---------------------------------------------------------
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index (‑1 ⇒ CPU).")

    # Model persistence ----------------------------------------------------------------
    parser.add_argument("--read_best_model", type=int, choices=[0, 1], default=0,
                        help="Load pre‑trained best model instead of training.")
    parser.add_argument("--store_model", type=int, choices=[0, 1], default=0,
                        help="Persist best model to disk after training.")

    # Optimisation hyper‑parameters -----------------------------------------------------
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini‑batch size.")
    parser.add_argument("--metrics", type=str, default="rmse", help="Primary metric.")

    # Verbosity -------------------------------------------------------------------------
    parser.add_argument("--show_process", type=int, choices=[0, 1], default=1,
                        help="Print per‑epoch metrics during training.")
    parser.add_argument("--show_setting", type=int, choices=[0, 1], default=1,
                        help="Display experiment configuration at start‑up.")

    # CEMS‑specific ---------------------------------------------------------------------
    parser.add_argument("--neigh_type", type=str, default="knn", choices=["random", "knnp", "knn"],
                        help="The method for neighbourhood selection: random, knnp (probabilistic), knn (nearest neighbours).")
    parser.add_argument("--cems_method", type=int, choices=[0, 1], default=1,
                        help="0 - Use point-wise estimation: cems_p, 1 - Use neighbourhood batching.")
    parser.add_argument("--input_sampling", type=int, choices=[0, 1], default=0,
                        help="Enable input sampling.")
    parser.add_argument("--manifold_sampling", type=int, choices=[0, 1], default=0,
                        help="Enable manifold sampling.")
    parser.add_argument("--sigma", type=float, default=1e-3, help="Base noise parameter for CEMS")
    parser.add_argument("--neigh_size", type=int, default=0, help="Neighbourhood size for point-wise estimation. If"
                                                                  "0, it will be defined as the smallest power of 2 that is larger than the ID.")


    return parser.parse_args()


# --------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------

def configure_device(gpu: int, show: bool = True) -> torch.device:
    """Select CPU / CUDA device depending on availability and user preference."""
    if torch.cuda.is_available() and gpu != -1:
        torch.cuda.set_device(gpu)
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")

    if show:
        logger.info("Using device: %s", device)
    return device


def enrich_arguments_with_dataset_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Merge dataset‑specific defaults into the argument namespace."""
    key = args.dataset if args.ts_name == "" else f"{args.dataset}-{args.ts_name}"
    args.dataset_name = key
    if key not in dataset_defaults:
        raise KeyError(f"Unknown dataset key '{key}'. Check config.dataset_defaults.")

    args.__dict__.update(dataset_defaults[key])
    return args


def prepare_output_dirs(root: Path, dataset_key: str) -> Tuple[Path, Path, Path]:
    """Ensure that result / ID / probability directories exist and return their paths."""
    result_root = root.expanduser().resolve()
    result_root.mkdir(parents=True, exist_ok=True)

    id_dir = result_root / "ids"
    probabilities_dir = result_root / "probabilities"
    dataset_dir = result_root / dataset_key

    for p in (id_dir, probabilities_dir, dataset_dir):
        p.mkdir(parents=True, exist_ok=True)

    return id_dir, probabilities_dir, dataset_dir





def apply_scaler_if_needed(scaler, y):
    """Safely fit/transform *y* with the given scaler (handles torch ⇆ numpy)."""
    if scaler is None:
        return y

    was_tensor = torch.is_tensor(y)
    y_np = y.cpu().detach().numpy() if was_tensor else y

    scaler.fit(y_np)
    y_scaled = scaler.transform(y_np)

    return torch.tensor(y_scaled) if was_tensor else y_scaled


def estimate_intrinsic_dim(
    args: argparse.Namespace,
    id_dir: Path,
    X: Any,
    Y: Any,
) -> None:
    """Populate *args.id* and *args.neigh_size* if intrinsic dimension is enabled."""

    logger.info("Estimating intrinsic dimension")

    id_name = f"{args.ts_name or args.dataset}_"
    id_name += "id.npy"

    id_path = id_dir / id_name
    args.id = get_id(str(id_path), X, Y)

    if args.neigh_size == 0:
        base_neigh = args.id + (args.id * (args.id + 1)) // 2
        args.neigh_size = shift_bit_length(base_neigh + 1)

    logger.info("Intrinsic dimension ≈ %d → neigh_size=%d", args.id, args.neigh_size)



# --------------------------------------------------------------------------------------
# Core workflow
# --------------------------------------------------------------------------------------

def train_and_evaluate(
    args: argparse.Namespace,
    device: torch.device,
    id_dir: Path,
    probabilities_dir: Path,
) -> Dict[str, float]:
    """Train CEMS (or load best model) and return key metrics for reporting."""
    # --------------------------------------------------------------------------
    # Data loading & pre‑processing
    # --------------------------------------------------------------------------
    data_packet, ts_data = load_data(args)
    X_train, Y_train = data_packet["x_train"], data_packet["y_train"]

    scaler = MinMaxScaler()
    Y_train = apply_scaler_if_needed(scaler, Y_train)
    data_packet["y_train"] = Y_train

    # ID estimation (updates args in‑place)
    estimate_intrinsic_dim(args, id_dir, X_train, Y_train)

    #Probability pre‑computation for mixup
    probs = get_probabilities(args, str(probabilities_dir), X_train, Y_train)

    # --------------------------------------------------------------------------
    # Train or load best model
    # --------------------------------------------------------------------------
    if args.read_best_model == 0:
        return _train_mode(args, data_packet, ts_data, device, scaler, probs)
    if args.read_best_model == 1:
        return _load_and_evaluate_best_model(args, data_packet, device, scaler)

    raise ValueError("--read_best_model must be 0 (train) or 1 (load best model)")


def _train_mode(
    args: argparse.Namespace,
    data_packet: Dict[str, Any],
    ts_data: Any,
    device: torch.device,
    scaler: Optional[Any],
    probs: np.ndarray,
) -> Dict[str, float]:
    """Train from scratch, evaluate and return metrics."""
    model = load_model(args, ts_data, device)
    logger.info("Untrained model initialised.")

    start_time = time.time()

    if args.cems_method:
        best_rmse, best_r, best_metric = algorithm.train_cems_batched(
            args, model, data_packet, probs, ts_data, scaler, device
        )
    else:
        best_rmse, best_r, best_metric = algorithm.train_cems(
            args, model, data_packet, probs, scaler, device
        )

    best_model_dict = {"rmse": best_rmse, "r": best_r}

    # Final evaluation on test split ---------------------------------------------------
    result = algorithm.test(
        args,
        model,
        best_model_dict[args.metrics],
        data_packet["x_test"],
        data_packet["y_test"],
        f"seed={args.seed}: Final test for best {args.metrics} model:\n",
        args.show_process,
        start_time,
        device,
        scaler=scaler,
    )

    algorithm.cal_worst_acc(
        args,
        data_packet,
        model,
        best_model_dict[args.metrics],
        result,
        start_time,
        device,
        scaler,
    )

    #Optional persistence
    if args.store_model:
        write_model(args, best_model_dict[args.metrics], args.result_root_path)

    # Neptune / stdout logging
    if args.run is not None:
        args.run[f"test_{args.metrics}"] = result[args.metrics]
        args.run[f"best_val_{args.metrics}"] = best_metric
        if args.metrics == "rmse":
            args.run["test_mape"] = result["mape"]

    return result


def _load_and_evaluate_best_model(
    args: argparse.Namespace,
    data_packet: Dict[str, Any],
    device: torch.device,
    scaler: Optional[Any],
) -> Dict[str, float]:
    """Load the stored best model (pickle) and verify its performance."""
    model_path = args.result_root_path / args.dataset / get_unique_file_name(args, "", ".pickle")
    logger.info("Loading best model from %s", model_path)

    with model_path.open("rb") as f:
        model = pickle.load(f)

    start_time = time.time()

    result = algorithm.test(
        args,
        model,
        data_packet["x_test"],
        data_packet["y_test"],
        f"seed={args.seed}: Final test for loaded best model.\n",
        True,
        start_time,
        device,
        scaler=scaler,
    )

    algorithm.cal_worst_acc(
        args,
        data_packet,
        model,
        result,
        start_time,
        device,
        scaler=scaler,
        run=args.run,
    )

    return result


# --------------------------------------------------------------------------------------
# Entry‑point glue logic
# --------------------------------------------------------------------------------------

def main() -> None:
    """CLI entry‑point"""
    args = parse_arguments()
    set_seed(args.seed)

    # Enrich args with dataset‑specific defaults ---------------------------------------
    args = enrich_arguments_with_dataset_defaults(args)

    # Device selection -----------------------------------------------------------------
    device = configure_device(args.gpu, show=bool(args.show_setting))

    # Output folders -------------------------------------------------------------------
    id_dir, probabilities_dir, dataset_dir = prepare_output_dirs(args.result_root_path, args.dataset)

    # Neptune run ----------------------------------------------------------------------
    args.run = None

    # Execute workflow -----------------------------------------------------------------
    metrics = train_and_evaluate(args, device, id_dir, probabilities_dir)

    logger.info("Final metrics → %s", metrics)

    if args.run is not None:
        args.run.stop()


if __name__ == "__main__":
    main()
