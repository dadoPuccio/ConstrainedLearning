import os
from datetime import datetime
import json

def init_logs_folder(savedir_base):
    exp_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    savedir = os.path.join(savedir_base, exp_date)

    os.makedirs(savedir, exist_ok=True)

    return savedir


def save_json(base_dir, fname, data):
    with open(os.path.join(base_dir, fname), "w") as json_file:
        json.dump(data, json_file, indent=4, sort_keys=True)


def make_log_record(phase, epoch, metrics, extra=None):
    d = {
        "phase": phase,
        "epoch": epoch,
        "train_loss": metrics["train"]["loss"],
        "train_acc": metrics["train"]["acc"],
        "train_recon_mse": metrics["train"]["recon_mse"],
        "train_violation": metrics["train"]["violation"],
        "train_satisfied": metrics["train"]["satisfied"],
        "test_loss": metrics["test"]["loss"],
        "test_acc": metrics["test"]["acc"],
        "test_recon_mse": metrics["test"]["recon_mse"],
        "test_violation": metrics["test"]["violation"],
        "test_satisfied": metrics["test"]["satisfied"],
    }
    if extra:
        # store train & test penalty weight separately
        d["train_penalty_weight"] = extra["penalty_weight"]
        d["test_penalty_weight"]  = extra["penalty_weight"]
    return d