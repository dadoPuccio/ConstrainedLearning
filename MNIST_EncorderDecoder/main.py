import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
import random
import pandas as pd

import argparse
import os
from copy import deepcopy

from utils.utils import evaluate
from utils.logs import *

from train_epoch_loops import *
from models.encoder_classifier import EncoderClassifier
from models.decoder import Decoder


def make_reproducible():
    seed = 123
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(dataset_path, batch_size, batch_size_eval):
    transform_train = transforms.Compose([
        transforms.ToTensor()
    ]) 

    transform_eval = transforms.Compose([
        transforms.ToTensor()
    ]) 

    train_dataset = datasets.MNIST(root=dataset_path, train=True, transform=transform_train, download=True)
    test_dataset  = datasets.MNIST(root=dataset_path, train=False, transform=transform_eval, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train_dataset_eval = datasets.MNIST(root=dataset_path, train=True, transform=transform_eval, download=True)
    train_loader_no_shuffle = DataLoader(train_dataset_eval, batch_size=batch_size_eval, shuffle=True)

    return train_loader, test_loader, train_loader_no_shuffle


# -----------------------------
# Warmup function
# -----------------------------
def warmup(model, loader, epochs):
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for ep in range(epochs):
        train_epoch_plain(model, loader, optimizer, device)
        metrics_train = evaluate(model, loader, device, VIOLATION_THRESHOLD, has_reconstruction_head=False)
        print(f"[Warmup] Epoch {ep+1}/{epochs} | train_loss={metrics_train['loss']:.4f}, acc={metrics_train['acc']:.4f}")

    return deepcopy(model.state_dict())

# -----------------------------
# Generic refinement runner
# -----------------------------
def refine(model, decoder, train_loader, test_loader, train_loader_no_shuffle, epochs, train_func, model_name="Model", penalty_start=None):

    optimizer_model = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.001)
    optimizer_decoder = None if decoder is None else optim.Adam(decoder.parameters(), lr=LEARNING_RATE, weight_decay=0.001)
    penalty_weight = penalty_start if penalty_start is not None else 0.0

    best_encoder = None
    best_decoder = None
    best_val = 1e9 if decoder is None else 0

    logs = []

    for ep in range(epochs):
        # Training
        if decoder is None:
            train_func(model, train_loader, optimizer_model, device)
        elif penalty_start is None:
            train_func(model, decoder, train_loader, optimizer_model, optimizer_decoder, RECON_FIXED_WEIGHT, device)
        else:
            train_func(model, decoder, train_loader, optimizer_model, optimizer_decoder, penalty_weight, VIOLATION_THRESHOLD, device)
            
        # Evaluate full train & test sets with decoder-aware evaluate()
        metrics_train = evaluate(model, train_loader_no_shuffle, device, VIOLATION_THRESHOLD, decoder=decoder, has_reconstruction_head=(decoder is not None))
        metrics_test  = evaluate(model, test_loader, device, VIOLATION_THRESHOLD, decoder=decoder, has_reconstruction_head=(decoder is not None))

        # extra info: penalty weight
        extra = {"penalty_weight": penalty_weight if penalty_start is not None else (RECON_FIXED_WEIGHT if decoder is not None else 0.0)}

        # log record
        record = make_log_record(epoch=ep, phase=model_name, metrics={"train": metrics_train, "test": metrics_test}, extra=extra)
        logs.append(record)

        if decoder is None:
            if metrics_train["loss"] < best_val:
                best_val = metrics_train["acc"]
                best_encoder = deepcopy(model.state_dict())
                best_decoder = None

        else:
            if metrics_train["satisfied"] > best_val:
                best_val = metrics_train["satisfied"]
                best_encoder = deepcopy(model.state_dict())
                best_decoder = deepcopy(decoder.state_dict())

        print(f"[{model_name}] Epoch {ep+1}/{epochs} | "
              f"train_loss={metrics_train['loss']:.4f}, acc={metrics_train['acc']:.4f}, "
              f"recon_mse={metrics_train['recon_mse']:.6f}, violation={metrics_train['violation']:.8f}, "
              f"satisfied={metrics_train['satisfied']:.8f}, "
              f"penalty_weight={extra['penalty_weight']:.4f}, "
              f"test_satisfied={metrics_test['satisfied']:.8f}")

        # Update penalty weight for next epoch if applicable
        if penalty_start is not None:
            penalty_weight = min(penalty_weight * PENALTY_GROWTH, MAX_PENALTY)

    df_logs = pd.DataFrame(logs)
    return df_logs, best_encoder, best_decoder


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--experiment', choices=['plain', 'fixed', 'penalty', 'all'])
    parser.add_argument('-wmp', '--warm_model_path', default='warm_models')
    parser.add_argument('-o', '--out_dir', default='logs')

    parser.add_argument('-pf', '--penalty_fixed', default=1., type=float)
    parser.add_argument('-pb', '--penalty_base', default=100., type=float)
    parser.add_argument('-pg', '--penalty_grow', default=1.01, type=float)

    args = parser.parse_args()

    BATCH_SIZE = 128
    BATCH_SIZE_EVAL = 1024

    WARMUP_EPOCHS = 5
    REFINE_EPOCHS = 250

    RECON_FIXED_WEIGHT = args.penalty_fixed
    PENALTY_BASE_WEIGHT = args.penalty_base
    PENALTY_GROWTH = args.penalty_grow

    VIOLATION_THRESHOLD = 1e-2
    LEARNING_RATE = 1e-3
    MAX_PENALTY = 1e4

    DATASET_PATH = "~/Datasets"

    MODEL_SPECS = {
        "latent_dim": 20,
        "hidden_dim": 256,
        "input_dim": 28*28,
        "num_classes": 10
    }

    JSON_VERSION_OF_CONSTANTS = {
        "BATCH_SIZE": BATCH_SIZE,
        "BATCH_SIZE_EVAL": BATCH_SIZE_EVAL,
        "WARMUP_EPOCHS": WARMUP_EPOCHS,
        "REFINE_EPOCHS": REFINE_EPOCHS, 
        "RECON_FIXED_WEIGHT":  RECON_FIXED_WEIGHT,
        "PENALTY_BASE_WEIGHT": PENALTY_BASE_WEIGHT,
        "PENALTY_GROWTH":   PENALTY_GROWTH,
        "VIOLATION_THRESHOLD": VIOLATION_THRESHOLD,
        "LEARNING_RATE": LEARNING_RATE,
        "MAX_PENALTY": MAX_PENALTY,
        "DATASET_PATH": DATASET_PATH,
        "MODEL_SPECS": MODEL_SPECS
    }

    make_reproducible()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    train_loader, test_loader, train_loader_no_shuffle = load_dataset(dataset_path=DATASET_PATH,
                                                                      batch_size=BATCH_SIZE,
                                                                      batch_size_eval=BATCH_SIZE_EVAL)


    if os.path.exists(os.path.join(args.warm_model_path, 'encoder_weights')):

        warm_model = EncoderClassifier(MODEL_SPECS['input_dim'], MODEL_SPECS['hidden_dim'], MODEL_SPECS['latent_dim'], MODEL_SPECS['num_classes']).to(device)
        warm_model.load_state_dict(torch.load(os.path.join(args.warm_model_path, 'encoder_weights')))
        
    else:
        os.makedirs(args.warm_model_path, exist_ok=True)
        base_model = EncoderClassifier(MODEL_SPECS['input_dim'], MODEL_SPECS['hidden_dim'], MODEL_SPECS['latent_dim'], MODEL_SPECS['num_classes']).to(device)
        warm_state_dict = warmup(base_model, train_loader, WARMUP_EPOCHS)
        torch.save(warm_state_dict, os.path.join(args.warm_model_path, 'encoder_weights'))

        warm_model = base_model

    savedir = init_logs_folder(args.out_dir)
    save_json(savedir, 'constants.json', JSON_VERSION_OF_CONSTANTS)

    if args.experiment in ['plain', 'all']: # 1) Plain refinement       
        df_plain, best_encoder, _ = refine(warm_model, None, train_loader, test_loader, train_loader_no_shuffle, REFINE_EPOCHS, train_epoch_plain, model_name="Plain")

        df_plain.to_csv(os.path.join(savedir,"df_plain.csv"), index=False)
        torch.save(best_encoder, os.path.join(savedir, 'plain_encoder_weights'))

    if args.experiment in ['fixed', 'all']: # 2) Fixed reconstruction refinement
        decoder_fixed = Decoder(MODEL_SPECS['latent_dim'], MODEL_SPECS['hidden_dim'], MODEL_SPECS['input_dim']).to(device)
        df_fixed, best_encoder, best_decoder = refine(warm_model, decoder_fixed, train_loader, test_loader, train_loader_no_shuffle, REFINE_EPOCHS, train_epoch_fixed, model_name="Fixed")

        df_fixed.to_csv(os.path.join(savedir,"df_fixed.csv"), index=False)
        torch.save(best_encoder, os.path.join(savedir, 'fixed_encoder_weights'))
        torch.save(best_decoder, os.path.join(savedir, 'fixed_decoder_weights'))
    
    if args.experiment in ['penalty', 'all']: # 3) Penalty reconstruction refinement
        decoder_penalty = Decoder(MODEL_SPECS['latent_dim'], MODEL_SPECS['hidden_dim'], MODEL_SPECS['input_dim']).to(device)
        df_penalty, best_encoder, best_decoder = refine(warm_model, decoder_penalty, train_loader, test_loader, train_loader_no_shuffle, REFINE_EPOCHS, train_epoch_penalty, model_name="Penalty", penalty_start=PENALTY_BASE_WEIGHT)

        df_penalty.to_csv(os.path.join(savedir,"df_penalty.csv"), index=False)
        torch.save(best_encoder, os.path.join(savedir, 'penalty_encoder_weights'))
        torch.save(best_decoder, os.path.join(savedir, 'penalty_decoder_weights'))