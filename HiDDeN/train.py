import os
import time
import torch
import numpy as np
import utils.utils as utils
import logging
from collections import defaultdict

from utils.options import *
from model.hidden import Hidden
from utils.average_meter import AverageMeter
from penalty_loss import *

def train(model: Hidden,
          device: torch.device,
          hidden_config: HiDDenConfiguration,
          train_options: TrainingOptions,
          this_run_folder: str,
          tb_logger):
    """
    Trains the HiDDeN model
    :param model: The model
    :param device: torch.device object, usually this is GPU (if avaliable), otherwise CPU.
    :param hidden_config: The network configuration
    :param train_options: The training settings
    :param this_run_folder: The parent folder for the current training run to store training artifacts/results/logs.
    :param tb_logger: TensorBoardLogger object which is a thin wrapper for TensorboardX logger.
                Pass None to disable TensorboardX logging
    :return:
    """
    if train_options.penalty_coefficient == None:
        penalty_loss = None
    else:
        penalty_loss = PenaltyLoss(ConstrainedPSNR(train_options.threshold, train_options.square), 
                                   penalty_coefficient=train_options.penalty_coefficient,
                                   penalty_increase_factor=train_options.increase_factor)
        if train_options.start_epoch >= train_options.increase_rate:
            for _ in range(train_options.start_epoch // train_options.increase_rate):
                penalty_loss.increase_penalty()

    train_data, val_data = utils.get_data_loaders(train_options.dataset_folder, hidden_config, train_options)
    file_count = len(train_data.dataset) # type: ignore
    if file_count % train_options.batch_size == 0:
        steps_in_epoch = file_count // train_options.batch_size
    else:
        steps_in_epoch = file_count // train_options.batch_size + 1

    print_each = 10
    images_to_save = 8
    saved_images_size = (512, 512)
    psnr_metric = pyiqa.create_metric('psnr', as_loss=True, device=device, loss_reduction='none')

    for epoch in range(train_options.start_epoch, train_options.number_of_epochs + 1):
        logging.info('\nStarting epoch {}/{}'.format(epoch, train_options.number_of_epochs))
        logging.info('Batch size = {}\nSteps in epoch = {}'.format(train_options.batch_size, steps_in_epoch))
        training_losses = defaultdict(AverageMeter)
        epoch_start = time.time()
        step = 1
        if epoch > 1 and penalty_loss != None:
            if penalty_loss.penalty_coefficient > 0 and epoch % train_options.increase_rate == 0 and not penalty_loss.is_max():
                penalty_loss.increase_penalty()
                logging.info(f"\nEpoch {epoch}: penalty coefficient increased to {penalty_loss.penalty_coefficient}\n")
        
        for image, _ in train_data:
            image = image.to(device)
            message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], hidden_config.message_length))).to(device)
            losses, (encoded_images, _, _) = model.train_on_batch([image, message], penalty_loss)
            
            if penalty_loss == None:
                batch_psnr = psnr_metric(image, encoded_images)
                losses['avg_psnr'] = torch.mean(batch_psnr).detach()

            for name, loss in losses.items():
                training_losses[name].update(loss)
            
            if step % print_each == 0 or step == steps_in_epoch:
                logging.info(
                    f'Epoch: {epoch}/{train_options.number_of_epochs} Step: {step}/{steps_in_epoch}\n')
                utils.log_progress(training_losses)
                logging.info('-' * 40)
            step += 1

        train_duration = time.time() - epoch_start
        logging.info('Epoch {} training duration {:.2f} sec'.format(epoch, train_duration))
        logging.info('-' * 40)
        utils.write_losses(os.path.join(this_run_folder, 'train.csv'), training_losses, epoch, train_duration)
        if tb_logger is not None:
            tb_logger.save_losses(training_losses, epoch)
            tb_logger.save_grads(epoch)
            tb_logger.save_tensors(epoch)

        first_iteration = True
        save_each = 100
        validation_losses = defaultdict(AverageMeter)
        logging.info('Running validation for epoch {}/{}'.format(epoch, train_options.number_of_epochs))

        below_threshold =  0
        for image, _ in val_data:
            image = image.to(device)
            message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], hidden_config.message_length))).to(device)
            losses, (encoded_images, _, _) = model.validate_on_batch([image, message], penalty_loss)
            batch_psnr = psnr_metric(image, encoded_images)
            if penalty_loss is not None:
                below_threshold += torch.count_nonzero(torch.nn.functional.relu(train_options.threshold - batch_psnr))
                
            else:
                losses['avg_psnr'] = torch.mean(batch_psnr).detach()
                
            for name, loss in losses.items():
                validation_losses[name].update(loss)
            if first_iteration and epoch % save_each == 0:
                if hidden_config.enable_fp16:
                    image = image.float()
                    encoded_images = encoded_images.float()
                utils.save_images(image.cpu()[:images_to_save, :, :, :],
                                  encoded_images[:images_to_save, :, :, :].cpu(),
                                  epoch,
                                  os.path.join(this_run_folder, 'images'), resize_to=saved_images_size)
                first_iteration = False


        utils.log_progress(validation_losses)
        logging.info('-' * 40)
        utils.write_losses(os.path.join(this_run_folder, 'validation.csv'), validation_losses, epoch,
                           time.time() - epoch_start)
        
        if epoch % save_each == 0:
            utils.save_checkpoint(model, train_options.experiment_name, epoch, os.path.join(this_run_folder, 'checkpoints'))
        

        
