import os
import pprint
import argparse
import torch
import pickle
import utils.utils as utils
import logging
import sys

from utils.options import *
from model.hidden import Hidden
from noise_layers.noiser import Noiser
from utils.noise_argparser import NoiseArgParser
from train import train
import numpy as np

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    parent_parser = argparse.ArgumentParser(description='Training of HiDDeN nets')
    subparsers = parent_parser.add_subparsers(dest='command', help='Sub-parser for commands')

    # NEW RUN
    new_run_parser = subparsers.add_parser('new', help='starts a new run')
    new_run_parser.add_argument('--data-dir', '-d', type=str, required=True,
                                help='The directory where the data is stored.')
    new_run_parser.add_argument('--batch-size', '-b', required=True, type=int, help='The batch size.')
    new_run_parser.add_argument('--epochs', '-e', default=200, type=int, help='Number of epochs to run the simulation.')
    new_run_parser.add_argument('--name', default='', type=str, help='The name of the experiment.')
    
    new_run_parser.add_argument('--size', '-s', default=224, type=int,
                                help='The size of the images (images are square so this is height and width).')
    new_run_parser.add_argument('--message', '-m', default=200, type=int, help='The length in bits of the watermark.')
    new_run_parser.add_argument('--bpp', default=0.0, type=float, help='BPP measure of the message. If not specified default message length is used.')
    new_run_parser.add_argument('--continue-from-folder', '-c', default='', type=str,
                                help='The folder from where to continue a previous run. Leave blank if you are starting a new experiment.')
    new_run_parser.add_argument('--enable-fp16', dest='enable_fp16', action='store_true',
                                help='Enable mixed-precision training.')

    new_run_parser.add_argument('--noise', nargs='*', action=NoiseArgParser,
                                help="Noise layers configuration. Use quotes when specifying configuration, e.g. 'cropout((0.55, 0.6), (0.55, 0.6))'")


    new_run_parser.add_argument('--penalty', nargs=3, 
                                help='Penalty Loss configurations. Specify penalty coefficient, increase factor and increase rate in sequence,' \
                                ' e.g. 0.1 1.5 5, meaning penalty has a starting coefficient of 0.1 increased of 50%% every 5 epochs')
    new_run_parser.add_argument('--PSNR', nargs='*', 
                                help='PSNR configuration. Specify PSNR penalty in sequence e.g. 20 True, meaning squared penalty enforces PSNR > 20. ' \
                                'If not square, second argument can be omitted ')

    new_run_parser.set_defaults(enable_fp16=False)


    # CONTINUE RUN
    continue_parser = subparsers.add_parser('continue', help='Continue a previous run')
    continue_parser.add_argument('--folder', '-f', required=True, type=str,
                                 help='Continue from the last checkpoint in this folder.')
    continue_parser.add_argument('--data-dir', '-d', required=False, type=str,
                                 help='The directory where the data is stored. Specify a value only if you want to override the previous value.')
    continue_parser.add_argument('--epochs', '-e', required=False, type=int,
                                help='Number of epochs to run the simulation. Specify a value only if you want to override the previous value.')

    args = parent_parser.parse_args()

    checkpoint = None
    loaded_checkpoint_file_name = None

    if args.command == 'continue':
        this_run_folder = args.folder
        options_file = os.path.join(this_run_folder, 'options-and-config.pickle')
        train_options, hidden_config, noise_config = utils.load_options(options_file)
        checkpoint, loaded_checkpoint_file_name = utils.load_last_checkpoint(os.path.join(this_run_folder, 'checkpoints'))
        train_options.start_epoch = checkpoint['epoch'] + 1
        if args.data_dir is not None:
            train_options.dataset_folder = args.data_dir
            train_options.train_folder = os.path.join(args.data_dir, 'train')
            train_options.validation_folder = os.path.join(args.data_dir, 'val')
        if args.epochs is not None:
            if train_options.start_epoch < args.epochs:
                train_options.number_of_epochs = args.epochs
            else:
                print(f'Command-line specifies of number of epochs = {args.epochs}, but folder={args.folder} '
                      f'already contains checkpoint for epoch = {train_options.start_epoch}.')
                exit(1)

    else:
        assert args.command == 'new'
        start_epoch = 1
        penalty_coeff = increase_fact = increase_rate = threshold = square = None
        name = args.name
        
        # check if penalty parameters have been properly specified
        if args.penalty is not None and args.PSNR is None:
            print("Specified penalty configuration with no penalty constraint function configuration")
            exit(1)
        if args.penalty is None and args.PSNR is not None:
            print("Specified penalty constraint function configuration with no penalty configuration")
            exit(1)
        
        if args.penalty is None and args.PSNR is None:
            print("Executing baseline training with no penalty")
            
        
        if args.penalty is not None and args.PSNR is not None:
            psnr_config = args.PSNR
            threshold = float(psnr_config[0])
            square = False
            if len(psnr_config) > 1:    # square option is present and needs to be converted from str to bool
                if psnr_config[1] == "True":
                    square = True
            
            penalty_coeff= float(args.penalty[0])
            increase_fact = float(args.penalty[1])
            increase_rate = int(args.penalty[2])

            if penalty_coeff < 0:
                print("Penalty Coefficient cannot be negative")
                exit(1)
            if increase_fact < 1:
                print("Penalty increase factor cannot be lower than 1")
                exit(1)
            if type(increase_rate) != int:
                print("Increase rate factor must be an integer")
                exit(1)
            
            suffix = f'coeff_{penalty_coeff}-factor_{increase_fact}-rate_{increase_rate}-PSNR_{threshold}'
            if name == '':
                name = suffix
            else:
                name = f'{name}-{suffix}'
            if square:
                name = f'{name}_square'

        if args.bpp != 0:
            message_length = int(np.floor(args.bpp * (args.size ** 2)))
        else:
            message_length = args.message

        print("Message length", message_length)
        train_options = TrainingOptions(
            batch_size=args.batch_size,
            number_of_epochs=args.epochs,
            penalty_coefficient=penalty_coeff,
            increase_factor=increase_fact,
            increase_rate=increase_rate,
            threshold=threshold,
            square=square,
            dataset_folder=args.data_dir,
            train_folder=os.path.join(args.data_dir, 'train'),
            validation_folder=os.path.join(args.data_dir, 'val'),
            runs_folder=os.path.join('.', 'runs'),
            start_epoch=start_epoch,
            experiment_name=name)

        noise_config = args.noise if args.noise is not None else []
        hidden_config = HiDDenConfiguration(H=args.size, W=args.size,
                                            message_length=message_length,
                                            encoder_blocks=4, encoder_channels=64,
                                            decoder_blocks=7, decoder_channels=64,
                                            use_discriminator=True,
                                            use_vgg=False,
                                            discriminator_blocks=3, discriminator_channels=64,
                                            decoder_loss=1,
                                            encoder_loss=0.7,
                                            adversarial_loss=1e-3,
                                            enable_fp16=args.enable_fp16
                                            )
            

        this_run_folder = utils.create_folder_for_run(train_options.runs_folder, name)
        with open(os.path.join(this_run_folder, 'options-and-config.pickle'), 'wb+') as f:
            pickle.dump(train_options, f)
            pickle.dump(noise_config, f)
            pickle.dump(hidden_config, f)
            

    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(this_run_folder, f'{train_options.experiment_name}.log')),
                            logging.StreamHandler(sys.stdout)
                        ])
   
    tb_logger = None

    noiser = Noiser(noise_config, device)
    model = Hidden(hidden_config, device, noiser, tb_logger)

    if args.command == 'continue':
        # if we are continuing, we have to load the model params
        assert checkpoint is not None
        logging.info(f'Loading checkpoint from file {loaded_checkpoint_file_name}')
        utils.model_from_checkpoint(model, checkpoint)

    logging.info('HiDDeN model: {}\n'.format(model.to_string()))
    logging.info('Model Configuration:\n')
    logging.info(pprint.pformat(vars(hidden_config)))
    logging.info('\nNoise configuration:\n')
    logging.info(pprint.pformat(str(noise_config)))
    logging.info('\nTraining train_options:\n')
    logging.info(pprint.pformat(vars(train_options)))

    train(model, device, hidden_config, train_options, this_run_folder, tb_logger)


if __name__ == '__main__':
    main()

