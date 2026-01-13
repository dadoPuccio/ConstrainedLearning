import torchvision
import torch
import torch.nn as nn
from torchvision.models.densenet import DenseNet121_Weights
from torchvision import transforms
from dataset import ChestXRayDataset
import numpy as np 
import csv
import copy
from sklearn.metrics import roc_auc_score
import utils.utils as utils
import os
from model.hidden import Hidden
from noise_layers.noiser import Noiser


class DenseNet121(nn.Module):
    """
    Densenet121 with additional classification layer for finetuning.

    """
    def __init__(self, out_size):
        """
        Args:
            out_size (int): Number of classes for classification.
        """
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Linear(num_ftrs, out_size)

    def forward(self, x):
        return self.densenet121(x)


def train_val_epoch(model, optimizer, train_dl, val_dl, epoch, device, frequency=25):
    """
    Train and validate model for a single epoch.

    Args:
        model: model to train.
        optimizer (Optimizer): optimizer to update model parameters. 
        train_dl (DataLoader): dataloader for training set.
        val_dl (Dataloader): dataloader for validation set.
        epoch (int): number of current epoch.
        device (str): Device to use to train model.
        frequency (int): frequency of output printing.

    """
    model.train()
    train_losses = []
    with torch.enable_grad():
        for i, data in enumerate(train_dl):
            imgs, labels = data
            if i % frequency == 0:
                print(f'Epoch-{epoch}: training batch {i}/{len(train_dl.dataset) // imgs.shape[0]}')

            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            train_loss = nn.functional.binary_cross_entropy_with_logits(preds, labels)
            train_loss.backward()
            optimizer.step()
            train_losses.append(train_loss.item())
    
    model.eval()
    val_losses = []
    with torch.no_grad():
        for i, data in enumerate(val_dl):
            imgs, labels = data
            if i % frequency == 0:
                print(f'Epoch-{epoch}: validation batch {i}/{len(val_dl.dataset) // imgs.shape[0]}')
            imgs = imgs.to(device)
            labels = labels.to(device)
            preds = model(imgs)
            val_losses.append(nn.functional.binary_cross_entropy_with_logits(preds, labels).item())
    

    return (np.mean(train_losses).item(), np.mean(val_losses).item())


def train_classifier(model_name, batch_size=64, epochs=250, lr=1e-4, out_path=None, dataset_dir=''):
    """
    Finetune Densenet121 classifier on Chest-XRay-14 dataset using Adam and 80/20 train/validation split.
    Training stops if validation loss doesn't improve for 10 consecutive epochs.

    Args:
        model_name (str): name used to save the model
        batch_size (int): batch size
        epochs (int): maximum number of epochs to train model
        lr (float): optimizer's learning rate 
    """
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor()
            ])
        
    train_set = ChestXRayDataset(dataset_dir, transform=transform, train=True, convert='RGB')
    n_classes = len(train_set.classes)

    train_set, val_set = torch.utils.data.random_split(train_set, [78484, 16818], generator=torch.Generator().manual_seed(42))

    train_dl = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=16)
    validation_dl = torch.utils.data.DataLoader(val_set, batch_size=batch_size, num_workers=12)
    
    model = DenseNet121(n_classes)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = np.inf
    epochs_since_improvement = 0
    best_model_state_dict = model.state_dict()

    loss_filename = f'{out_path}/{model_name}_train_val.csv'

    with open(loss_filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch', 'train_loss', 'val_loss'])

    for epoch in range(epochs):

        if epochs_since_improvement == 10:
            print(f"Epoch-{epoch}: Model hasn't improved in {epochs_since_improvement} epochs. Early stopping condition triggered")
            break

        train_loss, val_loss = train_val_epoch(model, optimizer, train_dl, validation_dl, epoch, device)

        improved = val_loss <= best_loss
        best_loss = min(val_loss, best_loss)

        if improved:
            epochs_since_improvement = 0
            best_model_state_dict = copy.deepcopy(model.state_dict())
        else:
            epochs_since_improvement += 1
            
        with open(loss_filename, 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch, '{:.4f}'.format(train_loss), '{:.4f}'.format(val_loss)])
            
    model.load_state_dict(best_model_state_dict)
    torch.save(model.state_dict(), f"{out_path}/{model_name}.pth")

    return model
    

def compute_AUROC(model, dataloader, save_dir, filename, encoder_decoder=None, message_length=30):
    """
    Compute AUROC score for each of the ChestXray-14 classes.

    Args:
        model: model to test
        dataloader (DataLoader): dataloader to test the model on.
        save_dir (str): directory path used to store the results' file.
        filename (str): name of file used to store results.
        encoder_decoder: encoder_decoder from Hidden model to watermark images to classify.
        message_lenght (int): watermark's lenght.

    """
    classes = np.array([ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'])
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.eval()
    model = model.to(device)
    frequency = 10

    if encoder_decoder is not None:
        encoder_decoder = encoder_decoder.to(device)
        encoder_decoder.eval()

    all_labels = torch.FloatTensor()
    all_preds = torch.FloatTensor().cuda()

    np.random.seed(42)
    with torch.no_grad():    
        for i, data in enumerate(dataloader):
            imgs, labels = data
            if i % frequency == 0:
                print(f'Test batch {i}/{len(dataloader.dataset) // imgs.shape[0]}')
            
            # accumulate all labels of the current dataset 
            all_labels = torch.cat((all_labels, labels), 0)
            imgs = imgs.to(device)
            # if encoder_decoder is present, it's used to watermark the images before the classification 
            if encoder_decoder is not None:
                messages = torch.Tensor(np.random.choice([0, 1], (imgs.shape[0], message_length))).to(device)
                encoded_imgs, _ , _ = encoder_decoder(imgs, messages)
                output = nn.functional.sigmoid(model(encoded_imgs))
            else:
                output = nn.functional.sigmoid(model(imgs))

            # accumulate all the model's prediction for the dataset
            all_preds = torch.cat((all_preds, output), 0)

    # to properly compute the AUROC scores we need to use the whole dataset 
    all_labels = all_labels.numpy()
    all_preds = all_preds.cpu().detach().numpy()

    with open(os.path.join(save_dir, f'{filename}-AUROC.csv'), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Pathology', 'AUROC'])
        # write score for each class
        for i in range(len(classes)):
            AUROC = roc_auc_score(all_labels[:, i], all_preds[:, i])
            writer.writerow([classes[i], '{:.4f}'.format(AUROC)])


if __name__ == '__main__':
    
    # ---------------------------- TRAIN CLASSIFIER ----------------------------

    batch_size = 64
    epochs = 250
    lr = 1e-4

    train = False
    dataset_dir = '' # SPECIFY DATASET DIR

    save_dir = 'HiDDen/runs_classifier'
    classifier_path = 'HiDDen/runs_classifier/No_Norm-ChestXRayDenseNet.pth'

    os.makedirs(save_dir, exist_ok=True)
    
    watermakers_directories = [
                                None,
                                # OTHER MODELS DIRS
                               ]

    transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor()
            ])
    
    test_dataset = ChestXRayDataset(dataset_dir, transform=transform, train=False, convert='RGB')
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=12)

    classes = test_dataset.classes
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    classifier = DenseNet121(len(classes))
    
    classifier = classifier.to(device)

    if train:
        classifier = train_classifier('No_Norm-ChestXRayDenseNet', batch_size, epochs, lr, out_path=save_dir, dataset_dir=dataset_dir)
    else:
        classifier.load_state_dict(torch.load(classifier_path, weights_only=True))

    for watermarker_path in watermakers_directories:
        
        if watermarker_path is not None:
            filename = watermarker_path.split('/')[-1].split(" ")[0]
            options_file = os.path.join(watermarker_path, 'options-and-config.pickle')
            train_options, hidden_config, noise_config = utils.load_options(options_file)
            checkpoint = utils.load_last_checkpoint(os.path.join(watermarker_path, 'checkpoints'))
            noiser = Noiser(noise_config, device)
            watermarker = Hidden(hidden_config, device, noiser, tb_logger=None)
            message_length = hidden_config.message_length 
            utils.model_from_checkpoint(watermarker, checkpoint)

            compute_AUROC(classifier, test_dl, save_dir, filename, watermarker.encoder_decoder, message_length)

        else:
            compute_AUROC(classifier, test_dl, save_dir, 'No_Norm-ChestXRayDenseNet')
        
    