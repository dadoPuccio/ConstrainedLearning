import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pyiqa
import torch
from PIL import Image
import torchvision.transforms as transforms
from matplotlib.gridspec import GridSpec

from utils import utils
from model.hidden import Hidden
from noise_layers.noiser import Noiser

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Plot Util')
    parser.add_argument('-p', '--plot', choices=['training_plots', 'ber', 'classification_table', 'visual_example'], required=True)

    args = parser.parse_args()

    plt.rcParams.update({
        "font.size": 16,          # base font size
        "axes.titlesize": 16,     # title
        "axes.labelsize": 16,     # x/y labels
        "xtick.labelsize": 12,    # x tick labels
        "ytick.labelsize": 12,    # y tick labels
        "legend.fontsize": 14,    # legend
        "figure.titlesize": 16    # figure title
    })

    save_directory = 'HiDDen/plots_hidden'
    os.makedirs(save_directory, exist_ok=True)

    if args.plot == 'training_plots':

        run_directories = [] # SPECIFY MODEL DIRS
        
        splits = ['train', 'validation']
        metrics = ['dec_mse', 'avg_psnr']
        count=0
        
        for metric in metrics:
            for split in splits:
                plt.figure(figsize=(6,5))

                for run in run_directories:
                    param_dict = utils.extract_training_params(run)
                    label = ''
                    if 'baseline' in param_dict.keys():
                        label = 'HiDDeN'
                    elif 'coeff' in param_dict.keys() and param_dict['fixed'] == False:
                            # label = fr'$\tau = {param_dict["coeff"]} - {int((float(param_dict["factor"]) - 1 )* 100)}\% \ \text{{increase every}} \ {int(param_dict["rate"])} - PSNR_{{\geq{int(param_dict["PSNR"])}}}$'
                            label = fr'$\text{{PSNR}}_{{\geq{int(param_dict["PSNR"])}}}$'
                    elif param_dict['fixed'] == True:
                            label = fr'Fixed $\tau = $ {param_dict["coeff"]} - $PSNR_{{\geq{int(param_dict["PSNR"])}}}$'

                    df = pd.read_csv(os.path.join(run, f'{split}.csv'))
                    if split == 'train':
                        plt.plot(df[metric], label=label)
                    else:
                        plt.plot(df[metric], label=label)
            
                plt.xlabel("Epoch")
                plt.xlim(0, 200)

                if metric == 'avg_psnr':
                    plt.ylabel(rf'$\text{{PSNR}}(I_{{co}}, I_{{en}})$')
                    plt.ylim(10, 60)
                    plt.title('Peak Signal-to-Noise Ratio - ' + split.capitalize() + ' Set')
                elif metric == 'dec_mse':
                    plt.ylabel(rf'$\text{{MSE}}(M_{{in}}, M_{{out}})$')
                    plt.ylim(0.1, 0.5)
                    plt.title('Message Loss - ' + split.capitalize() + ' Set')

                plt.grid(True)

                if count == 0:
                    plt.legend(loc="upper right")
                    count += 1
                plt.tight_layout()

                plt.savefig(os.path.join(save_directory, f"{metric}_{split}.pdf"))
                plt.close()


    if args.plot == 'ber':

        # ----------------- Boxplot  -----------------
        # First produce the .csv with the per-sample ber using compute_ber_test_set.py

        input_dir = '' # SPECIFY INPUT CSV DIR
        csv_files = os.listdir(input_dir)

        plt.figure(figsize=(6,5))
        for i, file in enumerate(utils.sorted_nicely(csv_files)):
            if '.csv' in file:
                param_dict = utils.extract_training_params(file)
                if 'baseline' in param_dict.keys():
                    label = 'HiDDeN'
                elif 'fixed' in param_dict.keys():
                    if param_dict['fixed'] == True:
                        label = f'PSNR$_{{\geq{int(param_dict["PSNR"])}}}$'
                    else:
                        label = f'PSNR$_{{\geq{int(param_dict["PSNR"])}}}$'
                else:
                    label = file.removesuffix(".cvs")
                data = np.loadtxt(os.path.join(input_dir, file), float, delimiter=',')
                plt.boxplot(data, sym='', positions=[i], tick_labels=[label])
            
        plt.ylabel('BER', labelpad=8)
        plt.ylim(0.1, 0.5)
        plt.title('Bit Error Rate - Test Set', pad=10)
        plt.tight_layout()
        plt.savefig(os.path.join(save_directory, 'Avg BER.pdf'), dpi=300)
        plt.close()

   
    if args.plot == 'classification_table':

        csv_files = [
            # SPECIFY INPUT AUROC FILES (TO BE PRODUCED BY train_classifier.py)
       ]

        rows = []
        row_names = ['Base Classifier', 'HiDDeN', r'$\mathrm{PSNR}_{\geq30}$', r'$\mathrm{PSNR}_{\geq40}$', r'$\mathrm{PSNR}_{\geq50}$']

        # Read each CSV and store values
        for i, f in enumerate(csv_files):
            df = pd.read_csv(f, header=None, skiprows=1, names=["Label", "Value"], dtype={"Value": float})
            row = df.set_index("Label")["Value"]
            row.name = row_names[i]
            rows.append(row)

        # Combine into a single DataFrame
        table_df = pd.DataFrame(rows)

        # Generate LaTeX table
        latex_table = table_df.transpose().to_latex(
            float_format="%.3f",
            index=True,
            caption="Results from Five Runs",
            label="tab:results"
        )

        print(latex_table)

    
    if args.plot == 'visual_example':

        # ----------------- Watermark Heatmap Plot -----------------

        run_directories = [] # SPECIFY MODEL DIRS
        
        input_image_paths = [] # SPECIFY EXAMPLE IMAGE
        
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        psnr_metric = pyiqa.create_metric('psnr', as_loss=True)
        
        transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor()
                ])
        
        np.random.seed(42)


        for img_id, image_path in enumerate(input_image_paths):
        
            img = transform(Image.open(image_path).convert('RGB')).unsqueeze_(0) # type: ignore
            img = img.to(device)
            message = torch.Tensor(np.random.choice([0, 1], (img.shape[0], 200))).to(device)
            minimum = 0
            maximum = 1

        
            fig = plt.figure(figsize=(12.5, 5))
            gs = GridSpec(2, 5, width_ratios=[1, 1, 1, 1, 1])

            axs = np.array([[fig.add_subplot(gs[i, j]) for j in range(5)] for i in range(2)])

            for ax in axs:
                for a in ax: 
                    a.axis("off")

            for i, model_dir in enumerate(run_directories):
                baseline = 'baseline' in model_dir
                options_file = os.path.join(model_dir, 'options-and-config.pickle')
                train_options, hidden_config, noise_config = utils.load_options(options_file)
                if not baseline:    
                    threshold = int(train_options.threshold)
                
                checkpoint = utils.load_last_checkpoint(os.path.join(model_dir, 'checkpoints'))               
                noiser = Noiser(noise_config, device)
                model = Hidden(hidden_config, device, noiser, tb_logger=None)
                utils.model_from_checkpoint(model, checkpoint)
                model.encoder_decoder.eval()
                
                with torch.no_grad():
                    encoded_img, noised_image , decoded_message = model.encoder_decoder(img, message)

                psnr = psnr_metric(img, encoded_img)
                ber = (torch.sum(torch.abs(message - decoded_message.round().clip(0, 1)), dim=1) / message.shape[1]).cpu().numpy().item()

                original = np.squeeze(torch.permute(transforms.transforms.F.rgb_to_grayscale(img), (0, 2, 3, 1)).cpu().numpy())

                if i == 0:
                    axs[0, 0].imshow(original, cmap='gray')
                    axs[0, 0].set_title('Original', fontsize=18)
                watermarked = np.squeeze(torch.permute(transforms.transforms.F.rgb_to_grayscale(encoded_img), (0, 2, 3, 1)).cpu().numpy())
                diff = ((watermarked - original))

                if baseline:
                    minimum = np.min(diff)
                    maximum = np.max(diff)
                
                watermark_cmap='seismic'

                axs[0, i+1].imshow(watermarked, cmap='gray')


                lim = max(np.abs(minimum), maximum)
                mappable = axs[1, i+1].imshow(diff, vmin=-lim, vmax=lim, cmap=watermark_cmap)        
            
                if baseline:
                    axs[0, i+1].set_title('HiDDeN', fontsize=18)
                else:
                    axs[0, i+1].set_title(rf'PSNR$_{{\geq{threshold}}}$', fontsize=18)
                
            cbar = fig.colorbar(
                    mappable,
                    ax=axs[1, 0],
                    location="right",
                    fraction=0.046,
                    pad=0.02
                )
            cbar.ax.yaxis.set_ticks_position("left")
            cbar.ax.yaxis.set_label_position("left")

            # Add label
            cbar.set_label("Pixel-wise difference\n from original", labelpad=8,  fontsize=12)

            fig.tight_layout()
            
            fig.savefig(os.path.join(save_directory, f'watermark{img_id}.png'), dpi=300, bbox_inches='tight')
            
            # plt.show()    
            plt.close()

