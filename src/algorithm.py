"""
algorithm.py – CEMS training / evaluation helpers
=================================================

Restored to the exact behaviour of the original codebase.
"""

from __future__ import annotations

import copy
import time
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from cems import get_batch_cems


# ----------------------------------------------------------------------------- #
# Evaluation helpers                                                            #
# ----------------------------------------------------------------------------- #
def _to_tensor(arr, device: torch.device) -> torch.Tensor:
    """Convert *arr* to float32 tensor on *device* (no-op if already)."""
    if isinstance(arr, np.ndarray):
        return torch.tensor(arr, dtype=torch.float32, device=device)
    if not arr.is_cuda and device.type == "cuda":
        return arr.to(device=device, dtype=torch.float32)
    return arr.to(dtype=torch.float32, device=device)


def cal_worst_acc(
    args,
    data_packet: Dict[str, Any],
    model: torch.nn.Module,
    best_model_state: dict,
    best_result_dict: Dict[str, float],
    all_begin: float,
    device: torch.device,
    scaler=None,
    run=None,
) -> None:
    """Compute worst-case metric across OOD splits (if enabled)."""
    if not getattr(args, "is_ood", False):
        return

    x_list = data_packet["x_test_assay_list"]
    y_list = data_packet["y_test_assay_list"]
    worst = 0.0 if args.metrics == "rmse" else 1e10

    for xs, ys in zip(x_list, y_list):
        res = test(
            args,
            model,
            best_model_state,
            xs,
            ys,
            "",
            False,
            all_begin,
            device,
            scaler,
        )
        acc = res[args.metrics]
        if args.metrics == "rmse":
            worst = max(worst, acc)
        else:  # r
            if abs(acc) < abs(worst):
                worst = acc

    print(f"worst {args.metrics} = {worst:.3f}")
    best_result_dict[f"worst_{args.metrics}"] = worst
    if run is not None:
        run[f"test_worst_{args.metrics}"] = worst


def test(
    args,
    model: torch.nn.Module,
    state_dict: dict,
    x_arr,
    y_arr,
    name: str,
    need_verbose: bool,
    epoch_start_time: float,
    device: torch.device,
    scaler=None,
) -> Dict[str, float]:
    """Run inference & compute mse / rmse / r / r² / mape."""
    model.load_state_dict(state_dict)
    model.eval()

    if args.dataset == "Dti_dg":
        val_iter = x_arr.shape[0] // args.batch_size
        val_len = args.batch_size
        y_arr = y_arr[: val_iter * val_len]
    else:
        val_iter, val_len = 1, x_arr.shape[0]

    preds: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(val_iter):
            xb = _to_tensor(x_arr[i * val_len : (i + 1) * val_len], device)
            preds.append(model(xb).cpu().numpy())

    y_pred = np.concatenate(preds, axis=0)
    if scaler is not None:
        y_pred = scaler.inverse_transform(y_pred)

    y_true = y_arr if isinstance(y_arr, np.ndarray) else y_arr.cpu().numpy()
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()

    mse = np.square(y_pred - y_true).mean()
    rmse = np.sqrt(mse)

    mean_p, std_p = y_pred.mean(0), y_pred.std(0)
    mean_g, std_g = y_true.mean(0), y_true.std(0)
    valid = std_g != 0
    corr = ((y_pred - mean_p) * (y_true - mean_g)).mean(0) / (std_p * std_g)
    corr = np.mean(corr[valid])

    non_zero = y_true != 0
    mape = (
        np.fabs(y_pred[non_zero] - y_true[non_zero]).mean()
        / np.fabs(y_true[non_zero]).mean()
        * 100
        if np.any(non_zero)
        else np.nan
    )

    res = {"mse": mse, "rmse": rmse, "r": corr, "r^2": corr**2, "mape": mape}

    if need_verbose:
        print(
            f"{name}corr={corr:.4f}, rmse={rmse:.4f}, mape={mape:.2f}% | "
            f"time={time.time() - epoch_start_time:.1f}s"
        )
    return res


def update_dict(
    args,
    model: torch.nn.Module,
    result: Dict[str, float],
    epoch: int,
    e_losses,
    best_mse: float,
    best_r2: float,
    best_state_mse: dict,
    best_state_r2: dict,
    best_summary: Dict[str, float],
) -> Tuple[float, float, dict, dict, Dict[str, float]]:
    """Track best MSE and R^2 across epochs."""
    if result["mse"] <= best_mse:
        best_mse = result["mse"]
        best_summary.update({"mse": best_mse, "rmse": result["rmse"]})
        best_state_mse = copy.deepcopy(model.state_dict())
        print(f"update best mse! epoch = {epoch}")

    if result["r"] ** 2 >= best_r2:
        best_r2 = result["r"] ** 2
        best_summary["r"] = result["r"]
        best_state_r2 = copy.deepcopy(model.state_dict())
        print(f"update best r! epoch = {epoch}")

    return best_mse, best_r2, best_state_mse, best_state_r2, best_summary


# ----------------------------------------------------------------------------- #
# Training loops                                                                #
# ----------------------------------------------------------------------------- #
def train_cems_batched(
    args,
    model: torch.nn.Module,
    data_packet: Dict[str, Any],
    probs=None,
    ts_data=None,
    scaler=None,
    device="cuda",
):
    """Anchor-based neighbourhood batching (identical to your original code)."""
    device = torch.device(device)
    model.train(True)
    optimizer = Adam(model.parameters(), args.lr)
    loss_fun = nn.MSELoss(reduction="mean").to(device)

    x_tr, y_tr = data_packet["x_train"], data_packet["y_train"]
    x_val, y_val = data_packet["x_valid"], data_packet["y_valid"]

    iteration = len(x_tr) // args.batch_size
    step_print_num = 30
    samples_idx = np.arange(x_tr.shape[0])

    best_state_mse = best_state_r2 = model.state_dict()
    best_sum = {"rmse": 1e10, "mse": 1e10, "r": 0.0, "r^2": 0.0}
    best_mse, best_r2 = 1e10, 0.0

    for epoch in range(args.num_epochs):
        model.train()
        epoch_start = time.time()
        shuffle_idx = np.random.permutation(samples_idx)
        idx_to_del: List[int] = []

        for idx in range(iteration):
            idx_1 = shuffle_idx[0]  # anchor
            probs_i = probs[idx_1] if args.neigh_type in ("knn", "knnp") else None

            # ---- neighbour selection (unchanged) ---------------------------------- #
            if args.neigh_type == "knnp":
                temp = probs_i[idx_to_del]
                probs_i[idx_to_del] = 0
                p_sum = probs_i.sum()
                idx_2 = np.random.choice(
                    samples_idx, size=args.batch_size - 1, replace=False, p=probs_i / p_sum
                )
                probs_i[idx_to_del] = temp
            elif args.neigh_type == "knn":
                idx_2 = probs_i[~np.isin(probs_i, idx_to_del)][1 : args.batch_size]
            else:  # random
                valid = samples_idx[~np.isin(samples_idx, idx_to_del)]
                idx_2 = np.random.choice(valid, size=min(args.batch_size, len(shuffle_idx)) - 1, replace=False)

            idx_all = np.insert(idx_2, 0, idx_1).astype(int)
            X = _to_tensor(x_tr[idx_all], device)
            Y = _to_tensor(y_tr[idx_all], device)

            if args.input_sampling:
                X, Y = get_batch_cems(args, X, Y)

            pred = model(X) if args.manifold_sampling == 0 else model.forward_mixup(args, X, Y)[0]

            if args.dataset == "TimeSereis":
                scale = ts_data.scale.expand(pred.size(0), ts_data.m)
                loss = loss_fun(pred * scale, Y * scale)
            else:
                loss = loss_fun(pred, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            idx_to_del.extend(idx_2.tolist() + [idx_1])
            shuffle_idx = shuffle_idx[~np.isin(shuffle_idx, idx_to_del)]

            # inline verbose for Dti_dg
            if (
                args.dataset == "Dti_dg"
                and (idx - 1) % max(1, iteration // step_print_num) == 0
                and args.show_process
            ):
                _ = test(
                    args,
                    model,
                    model.state_dict(),
                    x_val,
                    y_val,
                    f"Train epoch {epoch},  step={(epoch * iteration + idx)}:\t",
                    True,
                    epoch_start,
                    device,
                    scaler,
                )

        # ---- validation end-of-epoch ----------------------------------------------- #
        res = test(
            args,
            model,
            model.state_dict(),
            x_val,
            y_val,
            f"Train epoch {epoch}:\t",
            bool(args.show_process),
            epoch_start,
            device,
            scaler,
        )
        best_mse, best_r2, best_state_mse, best_state_r2, best_sum = update_dict(
            args,
            model,
            res,
            epoch,
            [],
            best_mse,
            best_r2,
            best_state_mse,
            best_state_r2,
            best_sum,
        )

    chosen = best_state_mse if args.metrics == "rmse" else best_state_r2
    return best_state_mse, best_state_r2, best_sum[args.metrics]


def train_cems(
    args,
    model: torch.nn.Module,
    data_packet: Dict[str, Any],
    probs=None,
    scaler=None,
    device="cuda",
):
    """Original per-sample neighbourhood construction loop."""
    device = torch.device(device)
    model.train(True)
    optimizer = Adam(model.parameters(), args.lr)
    loss_fun = nn.MSELoss(reduction="mean").to(device)

    x_tr, y_tr = data_packet["x_train"], data_packet["y_train"]
    x_val, y_val = data_packet["x_valid"], data_packet["y_valid"]

    iteration = len(x_tr) // args.batch_size
    step_print_num = 30
    samples_idx = np.arange(x_tr.shape[0])

    best_state_mse = best_state_r2 = model.state_dict()
    best_sum = {"rmse": 1e10, "mse": 1e10, "r": 0.0, "r^2": 0.0}
    best_mse, best_r2 = 1e10, 0.0

    for epoch in range(args.num_epochs):
        model.train()
        epoch_start = time.time()
        shuffle_idx = np.random.permutation(samples_idx)

        for idx in range(iteration):
            if args.batch_neigh:
                idx_1 = shuffle_idx[idx]
            else:
                idx_1 = shuffle_idx[idx * args.batch_size : (idx + 1) * args.batch_size]

            if args.mixtype in ("knnp", "knn"):
                probs_idx = probs[idx_1]
                if args.mixtype == "knnp":
                    idx_2 = np.array(
                        [np.random.choice(len(r), size=args.neigh_size, p=r, replace=False) for r in probs_idx]
                    )
                else:
                    idx_2 = probs_idx[:, 1 : args.neigh_size + 1]
            else:
                idx_2 = np.array(
                    [
                        np.random.choice(samples_idx, size=args.neigh_size, replace=False)
                        for _ in range(args.batch_size)
                    ]
                )

            X_i = _to_tensor(x_tr[idx_1], device)
            Y_i = _to_tensor(y_tr[idx_1], device)
            X_k = _to_tensor(x_tr[idx_2], device)
            Y_k = _to_tensor(y_tr[idx_2], device)

            X_mix, Y_mix = (X_i, Y_i)
            if args.input_sampling:
                X_mix, Y_mix = get_batch_cems(args, X_i, Y_i, X_k, Y_k)

            pred = (
                model(X_mix)
                if args.manifold_sampling == 0
                else model.forward_mixup(args, X_mix, Y_mix, X_k, Y_k)[0]
            )

            loss = loss_fun(pred, Y_mix)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (
                args.dataset == "Dti_dg"
                and (idx - 1) % max(1, iteration // step_print_num) == 0
                and args.show_process
            ):
                _ = test(
                    args,
                    model,
                    model.state_dict(),
                    x_val,
                    y_val,
                    f"Train epoch {epoch}, step={(epoch * iteration + idx)}:\t",
                    True,
                    epoch_start,
                    device,
                    scaler,
                )

        res = test(
            args,
            model,
            model.state_dict(),
            x_val,
            y_val,
            f"Train epoch {epoch}:\t",
            bool(args.show_process),
            epoch_start,
            device,
            scaler,
        )
        best_mse, best_r2, best_state_mse, best_state_r2, best_sum = update_dict(
            args,
            model,
            res,
            epoch,
            [],
            best_mse,
            best_r2,
            best_state_mse,
            best_state_r2,
            best_sum,
        )

    return best_state_mse, best_state_r2, best_sum[args.metrics]
