""" Compute BER on ChestXRayDataset test set for the all the watermarking methods (HiDDeN, PSNR 30, 40, 50)"""

import torch
from torchvision import transforms
from dataset import ChestXRayDataset
import numpy as np 
import csv
import utils.utils as utils
import os
from model.hidden import Hidden
from noise_layers.noiser import Noiser


if __name__ == '__main__':

    dataset_dir = '' # SPECIFY DATASET DIR
    model_directories = [] # SPECIFY MODEL DIR
    base_ber_folder = 'HiDDen/BERs' # SPECIFY OUTPUT DIR

    batch_size = 32
    np.random.seed(42)

    transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor()
            ])
    
    test_dataset = ChestXRayDataset(dataset_dir, transform=transform, train=False, convert='RGB')
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=12)

    classes = test_dataset.classes
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    
    if not os.path.exists(base_ber_folder):
        os.makedirs(base_ber_folder)

    for model_dir in model_directories:
        
        print("Current model dir: ", model_dir)
        options_file = os.path.join(model_dir, 'options-and-config.pickle')
        train_options, hidden_config, noise_config = utils.load_options(options_file)

        ## LOAD MODEL
        checkpoint = utils.load_last_checkpoint(os.path.join(model_dir, 'checkpoints'))   
        # checkpoint = utils.load_best_checkpoint(model_dir)              
                    
        noiser = Noiser(noise_config, device)
        model = Hidden(hidden_config, device, noiser, tb_logger=None)
        utils.model_from_checkpoint(model, checkpoint)
        model.encoder_decoder.eval()
        
        ## test 
        filename = model_dir.split('/')[-1] + '.csv' 
        step = 0
        ber_results = []  # for current model
        for image, _ in test_dl:
            step += 1
            image = image.to(device)
            message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], hidden_config.message_length))).to(device)

            # compute BER
            with torch.no_grad():
                encoded_img, noised_image, decoded_message = model.encoder_decoder(image, message)
            
            ber = torch.sum(torch.abs(message - decoded_message.round().clip(0, 1)), dim=1) / message.shape[1]
            ber_results = ber_results + [b.item() for b in ber]

            if step % 10 == 0:
                print(f'Step {step * batch_size}/{len(test_dataset)}    {np.mean(ber_results)}')
        
        with open(os.path.join(base_ber_folder, filename), "w", newline="") as f:
            writer = csv.writer(f)
            for n in ber_results:
                writer.writerow([n])



