import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse

import pandas as pd
import torch
import matplotlib.pyplot as plt

from plot_utils import read_json, revive_model, plot_metric, summary_table, plot_density
from main import load_dataset

parser = argparse.ArgumentParser()

parser.add_argument('-dirs', '--log_dirs', nargs='+')
parser.add_argument('-o', '--out_dir', default='plots')

args = parser.parse_args()

print(args)

os.makedirs(args.out_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataframes = []
model_types = []
encoders = []
decoders = []
json_of_constants = []

counter = 0

for input_dir in args.log_dirs:

    current_json_of_constants = read_json(os.path.join(input_dir, "constants.json"))

    for model_type in ['plain', 'fixed', 'penalty']:

        if os.path.exists(os.path.join(input_dir, f"df_{model_type}.csv")):
            dataframes.append(pd.read_csv(os.path.join(input_dir, f"df_{model_type}.csv")))
            json_of_constants.append(current_json_of_constants)

            encoder, decoder = revive_model(input_dir, model_type, current_json_of_constants, device)

            encoders.append(encoder)
            decoders.append(decoder)

            model_types.append(model_type)

            counter += 1
    
metrics_to_plot = ["loss", "acc", "recon_mse", "violation", "penalty_weight", "satisfied"]

for m in metrics_to_plot:
    
    labels = []
    for json_dict, model_type in zip(json_of_constants, model_types):
        if model_type == 'plain':
            labels.append("Plain")
        elif model_type == 'fixed':
            labels.append("Fixed_" + str(json_dict["RECON_FIXED_WEIGHT"]))
        elif model_type == 'penalty':
            labels.append("Penalty_" + str(json_dict["PENALTY_BASE_WEIGHT"]) + "_" + str(json_dict["PENALTY_GROWTH"]))

    plot_metric(dataframes, labels, m, f"{m} vs Epoch", args.out_dir, phase="train")
    plot_metric(dataframes, labels, m, f"{m} vs Epoch", args.out_dir, phase="test")


_, test_loader, train_loader_no_shuffle = load_dataset(json_of_constants[0]['DATASET_PATH'], json_of_constants[0]['BATCH_SIZE'], json_of_constants[0]['BATCH_SIZE_EVAL'])


# Summary table (final train + test metrics)
df_summary = summary_table(encoders, decoders, json_of_constants, labels, train_loader_no_shuffle, test_loader, device)
print("\nSummary Table:")
print(df_summary)
print()

def transform_value(x):
    parts = x.split("_")
    
    # your custom logic here
    if len(parts) == 2:
        return parts[0] + " ($\\lambda = " + parts[1] + "$)"
    elif len(parts) == 3:
        return parts[0] + " ($\\tau_0 = " + parts[1] + "$, $\\gamma = " + parts[2] + "$)"
    else: 
        return x

df_latex = df_summary.copy()
df_latex["Model"] = df_latex["Model"].apply(transform_value)
# print(df_latex.to_latex(index=False))

labels_plots = [v for v in df_latex["Model"]]

for e, d, json_dict, name in zip(encoders, decoders, json_of_constants, labels_plots):

    plot_density(e, d, train_loader_no_shuffle, device, json_dict['VIOLATION_THRESHOLD'], name, 'Train', args.out_dir)
    plot_density(e, d, test_loader, device, json_dict['VIOLATION_THRESHOLD'], name, 'Test', args.out_dir)


for m in metrics_to_plot:

    plot_metric(dataframes, labels_plots, m, f"{m} vs Epoch", args.out_dir, phase="train")
    plot_metric(dataframes, labels_plots, m, f"{m} vs Epoch", args.out_dir, phase="test")

