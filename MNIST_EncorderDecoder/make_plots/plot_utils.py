import os
from models.decoder import Decoder
from models.encoder_classifier import EncoderClassifier
import torch
import json
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from utils.utils import reconstruction_error


from utils.utils import evaluate, flatten


plt.rcParams.update({
    "font.size": 16,          # base font size
    "axes.titlesize": 16,     # title
    "axes.labelsize": 16,     # x/y labels
    "xtick.labelsize": 12,    # x tick labels
    "ytick.labelsize": 12,    # y tick labels
    "legend.fontsize": 14,    # legend
    "figure.titlesize": 16    # figure title
})

def read_json(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def revive_model(path, model_type, json_of_constants, device):

    encoder = EncoderClassifier(json_of_constants['MODEL_SPECS']['input_dim'], json_of_constants['MODEL_SPECS']['hidden_dim'], json_of_constants['MODEL_SPECS']['latent_dim'], json_of_constants['MODEL_SPECS']['num_classes']).to(device)
    print(os.path.join(path, f'{model_type}_encoder_weights'), os.path.exists(os.path.join(path, f'{model_type}_encoder_weights')))
    encoder.load_state_dict(torch.load(os.path.join(path, f'{model_type}_encoder_weights')))

    if os.path.exists(os.path.join(path, f'{model_type}_decoder_weights')):
        decoder = Decoder(json_of_constants['MODEL_SPECS']['latent_dim'], json_of_constants['MODEL_SPECS']['hidden_dim'], json_of_constants['MODEL_SPECS']['input_dim']).to(device)
        decoder.load_state_dict(torch.load(os.path.join(path, f'{model_type}_decoder_weights')))
    else: 
        decoder = None

    return encoder, decoder


def revive_models(path, json_of_constants, device):

    model_plain = EncoderClassifier(json_of_constants['MODEL_SPECS']['input_dim'], json_of_constants['MODEL_SPECS']['hidden_dim'], json_of_constants['MODEL_SPECS']['latent_dim'], json_of_constants['MODEL_SPECS']['num_classes']).to(device)
    model_plain.load_state_dict(torch.load(os.path.join(path, 'plain_encoder_weights')))

    model_fixed = EncoderClassifier(json_of_constants['MODEL_SPECS']['input_dim'], json_of_constants['MODEL_SPECS']['hidden_dim'], json_of_constants['MODEL_SPECS']['latent_dim'], json_of_constants['MODEL_SPECS']['num_classes']).to(device)
    model_fixed.load_state_dict(torch.load(os.path.join(path, 'fixed_encoder_weights')))
    decoder_fixed = Decoder(json_of_constants['MODEL_SPECS']['latent_dim'], json_of_constants['MODEL_SPECS']['hidden_dim'], json_of_constants['MODEL_SPECS']['input_dim']).to(device)
    decoder_fixed.load_state_dict(torch.load(os.path.join(path, 'fixed_decoder_weights')))

    model_penalty = EncoderClassifier(json_of_constants['MODEL_SPECS']['input_dim'], json_of_constants['MODEL_SPECS']['hidden_dim'], json_of_constants['MODEL_SPECS']['latent_dim'], json_of_constants['MODEL_SPECS']['num_classes']).to(device)
    model_penalty.load_state_dict(torch.load(os.path.join(path, 'penalty_encoder_weights')))
    decoder_penalty = Decoder(json_of_constants['MODEL_SPECS']['latent_dim'], json_of_constants['MODEL_SPECS']['hidden_dim'], json_of_constants['MODEL_SPECS']['input_dim']).to(device)
    decoder_penalty.load_state_dict(torch.load(os.path.join(path, 'penalty_decoder_weights')))
    
    return model_plain, model_fixed, decoder_fixed, model_penalty, decoder_penalty
    

def plot_metric(df_list, labels, metric, title, out_dir, phase):

    plt.figure(figsize=(6,5))
    for df, label in zip(df_list, labels):
        if label == "Plain":
            label = 'Classification Only'
            if metric == 'satisfied':
                continue
        if "Penalty" in label:
            zorder=3
        else:
            zorder=2
        plt.plot(df["epoch"], df[f"{phase}_{metric}"], label=label, zorder=zorder)

    plt.xlabel("Epoch")
    plt.xlim(0, 250)

    if metric == 'acc':
        metric = 'Accuracy'
        plt.ylim(0.8, 1)
    elif metric == 'satisfied':
        metric = metric.capitalize()
        plt.ylim(0.3, 0.8)
    else:
        metric = metric.capitalize()
        
    plt.ylabel(phase.capitalize() + " " + metric)
    # plt.title(f"{phase.capitalize() + " " + metric} vs Epoch")
    plt.grid(True)
    # if metric == "Satisfied":
    #     plt.legend(loc="center right")
    # else:
    plt.legend(loc="lower right")
    plt.tight_layout()

    plt.savefig(os.path.join(out_dir, f"{phase}_{metric}.pdf"))
    plt.close()


def summary_table(encoders, decoders, json_of_constants, names, train_loader, test_loader, device):
    
    rows = []
    for e, d, json_dict, n in zip(encoders, decoders, json_of_constants, names):

        tr = evaluate(e, train_loader, device, json_dict['VIOLATION_THRESHOLD'], decoder=d, has_reconstruction_head=(d is not None))
        te = evaluate(e, test_loader, device, json_dict['VIOLATION_THRESHOLD'], decoder=d, has_reconstruction_head=(d is not None))

        if d is None:
            final_pen = 0.0
        elif n=="Fixed":
            final_pen = json_dict["RECON_FIXED_WEIGHT"]
        else:
            final_pen = min(json_dict["PENALTY_BASE_WEIGHT"] * (json_dict["PENALTY_GROWTH"] ** json_dict["REFINE_EPOCHS"]), json_dict["MAX_PENALTY"])
        rows.append({
            "Model": n,
            "Train Loss": tr["loss"],
            "Train Acc": tr["acc"],
            "Train Recon MSE": tr["recon_mse"],
            "Train Violation": tr["violation"],
            "Train Satisfied": tr["satisfied"],
            # "Train Penalty": final_pen,
            "Test Loss": te["loss"],
            "Test Acc": te["acc"],
            "Test Recon MSE": te["recon_mse"],
            "Test Violation": te["violation"],
            "Test Satisfied": te["satisfied"],
            # "Test Penalty": final_pen,
        })

    return pd.DataFrame(rows)


def plot_confusion(model, loader, title, out_dir, device):
    y_true = []; y_pred = []
    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            flat = flatten(imgs)
            out = model(flat)
            preds = out["logits"].argmax(dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title(title)

    plt.savefig(os.path.join(out_dir, f"confusion_{title}.pdf"))
    plt.close()



def show_reconstructions(model, decoder, loader, device, out_dir, n=8, title="Train"):
    imgs, _ = next(iter(loader))
    imgs = imgs[:n].to(device)
    flat = flatten(imgs)
    with torch.no_grad():
        out = model(flat)
        if decoder:
            recon = decoder(out["z"]).view_as(imgs)
        else:
            recon = torch.zeros_like(imgs)
    fig, axes = plt.subplots(2, n, figsize=(n*2,4))
    for i in range(n):
        axes[0,i].imshow(imgs[i,0].cpu(), cmap="gray"); axes[0,i].axis("off")
        axes[1,i].imshow(recon[i,0].cpu(), cmap="gray"); axes[1,i].axis("off")
    plt.suptitle(f"{title} - {type(model).__name__} (Top: original, Bottom: reconstructed)")
    
    plt.savefig(os.path.join(out_dir, f"reconstruction_{title.replace(" ","_")}.pdf"))
    plt.close()


def plot_density(model, decoder, loader, device, violation_threshold, title, phase, out_dir, n_bins=200, multiplier=1.):

    model.eval()
    if decoder:
        decoder.eval()
        has_reconstruction_head=True
    else:
        has_reconstruction_head=False

    count_satisfied = 0
    recon_mse_list = []
    violation_list = []
    total_count = 0
    

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        flat = flatten(imgs)

        total_count += imgs.shape[0]

        out = model(flat)

        # reconstruction
        if has_reconstruction_head and decoder is not None:
            recon = decoder(out["z"]).view_as(imgs)
        else:
            recon = torch.zeros_like(imgs)  # for plain model

        mse_per_img = reconstruction_error(imgs, recon).cpu().detach().numpy()
        count_satisfied += np.sum(mse_per_img <= violation_threshold * multiplier)
        recon_mse_list.extend(mse_per_img)

        # violation metric for logging (not squared)
        v = np.maximum(0, mse_per_img - violation_threshold)
        violation_list.extend(v)

    bin_width = 0.0002  
    bins = np.arange(0, 0.1, bin_width)

    plt.figure(figsize=(6,5))
    plt.xlim(0, 0.04)
    plt.ylim(0, 150)
    plt.hist(recon_mse_list, bins=bins, density=True)
    plt.axvline(x=violation_threshold, linestyle='--', color="red", label=r"Threshold $\theta$")
    # plt.axvline(x=violation_threshold * multiplier, linestyle='--', color="red", label=r"Threshold $\times$ "+ f"{multiplier}" )
    plt.xlabel("Reconstruction Loss")
    plt.ylabel("Density")
    plt.title(f"{title} - {phase}") # (Satisfied={count_satisfied/total_count:.4f})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(out_dir, f"Density_{title.split(" ")[0]}_{phase}.pdf"))
    plt.close()