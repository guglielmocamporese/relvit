##################################################
# Imports
##################################################

import sys
import subprocess
import os
import logging
import pytorch_lightning as pl

# custom
from utils import get_args, get_trainer
from datasets.dataloaders import get_dataloaders
from models import relvit

# settings
format_str = '%(asctime)s - [%(levelname)s] - %(message)s'
logging.basicConfig(level=logging.INFO, format=format_str, datefmt='%d-%b-%y %H:%M:%S')


# main function
def main(args):
    import torch

    # logging
    if args.num_gpus > 0:
        nvidia_smi = subprocess.Popen(['nvidia-smi'], stdout=subprocess.PIPE, 
                                      stderr=subprocess.PIPE).communicate()[0].decode('utf8')
        logging.info(nvidia_smi)
    else:
        logging.info('Using CPU!')
    hostname = os.uname().nodename # Node name
    logging.info(f'hostname:{hostname}')
    
    # dataloaders, model, and trainer
    dls = get_dataloaders(args)
    model = relvit.get_model(args)
    trainer = get_trainer(args, dls)

    # train, val, test mode
    if args.mode in ['train', 'training']:
        trainer.fit(model, dls['train_aug'], dls['validation'])
        trainer.validate(model=None, val_dataloaders=dls['validation'])
    elif args.mode in ['validate', 'validation', 'validating']:
        trainer.validate(model, val_dataloaders=dls['validation'])
    elif args.mode in ['train_linear']:
        trainer.fit(model, dls['train_aug'], dls['validation'])
        trainer.validate(model, val_dataloaders=dls['validation'])
    elif args.mode in ['validate_lienar', 'validation_linear', 'validating_linear']:
        trainer.validate(model, val_dataloaders=dls['validation'])
    else:
        raise Exception(f'Error. Model "{args.mode}" not supported.')


##################################################
# Main
##################################################

if __name__ == '__main__':

    # parse inputs args
    args = get_args(sys.stdin, verbose=True)
    pl.seed_everything(args.seed, workers=True)

    # run main
    main(args)