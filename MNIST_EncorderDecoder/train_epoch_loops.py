from torch import nn
import torch
from utils.utils import flatten
import numpy as np

def train_epoch_plain(model, loader, optimizer, device):
    model.train()
    ce = nn.CrossEntropyLoss()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        flat = flatten(imgs)
        out = model(flat)
        loss = ce(out["logits"], labels)
        loss.backward()
        optimizer.step()

def train_epoch_fixed(model, decoder, loader, optimizer_model, optimizer_decoder, alpha, device):
    model.train(); decoder.train()
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer_model.zero_grad(); optimizer_decoder.zero_grad()
        flat = flatten(imgs)
        out = model(flat)
        recon = decoder(out["z"])

        loss = ce(out["logits"], labels) + alpha*mse(recon, flat)
        loss.backward()
        optimizer_model.step(); optimizer_decoder.step()

def train_epoch_penalty(model, decoder, loader, optimizer_model, optimizer_decoder, penalty_weight, violation_threshold, device):
    model.train(); decoder.train()
    ce = nn.CrossEntropyLoss(reduction="none")
    mse_per_pixel = nn.MSELoss(reduction="none")
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer_model.zero_grad(); optimizer_decoder.zero_grad()
        flat = flatten(imgs)
        out = model(flat)
        recon = decoder(out["z"])

        per_img_mse = mse_per_pixel(recon, flat).mean(dim=1)

        loss = ce(out["logits"], labels) + penalty_weight * torch.clamp(per_img_mse - violation_threshold, min=0.0) 
        loss = loss.mean() 
        loss.backward()
        optimizer_model.step(); optimizer_decoder.step()
