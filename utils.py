##################################################
# Imports
##################################################

import os
import math
import tarfile
import requests
import logging
import argparse
import json
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb


def get_args(stdin, verbose=False):
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser(stdin)

    # global params
    parser.add_argument('--seed', type=int, default=35771, help='The random seed.')
    parser.add_argument('--logger', type=str, default='wandb', help='The logger to use for the experiments.')
    parser.add_argument('--mode', type=str, default='train', help='The mode of the program, can "train" or "validate"')
    parser.add_argument('--num_gpus', type=int, default=1, help='The number of GPUs.')
    parser.add_argument('--epochs', type=int, default=100, help='The number of epochs for the training.')
    parser.add_argument('--task', type=str, default='upstream', 
                        help='The name of the task, can be "upstream" or "downstream"')
    parser.add_argument('--exp_id', type=str, default='', 
                        help='Experiment ID (like "u1").')

    # datasets params
    parser.add_argument('--dataset', type=str, default='imagenet', help='Name of the dataset.')
    parser.add_argument('--data_base_path', type=str, default='./datasets/data', help='The data base path.')
    parser.add_argument('--labels_path', type=str, default='', help='The path of the txt file containing imagenet labels subset.')
    parser.add_argument('--batch_size', type=int, default=256, help='The batch size.')
    parser.add_argument('--num_workers', type=int, default=8, help='The number of workers for the dataloader.')
    parser.add_argument('--debug_dataset', dest='debug_dataset', action='store_true', 
                        help='Print useful info about the dataset.')
    parser.add_argument('--shuffle_patches', dest='shuffle_patches', action='store_true', 
                        help='Shuffle the input patches.')

    # optimizer params
    parser.add_argument('--optimizer', type=str, default='adamw', help='The kind of optimizer.', 
                        choices=['sgd', 'adam', 'adamw'])
    parser.add_argument('--lr', type=float, default=5e-4, help='The learning rate.')
    parser.add_argument('--metric_monitor', type=str, default='val_acc_rel', 
                        help='The metric used for early stopping.')
    parser.add_argument('--grad_clip_val', type=float, default=1.0, help='Gradient clipping value.')

    ## model params
    # model setting
    parser.add_argument('--backbone', type=str, default='vit', help='Backbone of the model.', 
                        choices=['vit', 'swin', 't2t_vit'])
    parser.add_argument('--model_size', type=str, default='small', help='Backbone of the model.')
    parser.add_argument('--model_checkpoint', type=str, default='', help='The model checkpoint path (*.ckpt).')
    parser.add_argument('--patch_size', type=int, default=-1, help='The patch size.')

    # tasks setting
    parser.add_argument('--use_relations', dest='use_relations', action='store_true', 
                        help='Learn patches spatial relations.')
    parser.add_argument('--use_supervision', dest='use_supervision', action='store_true', help='Learn image classes.')
    parser.add_argument('--use_distances', dest='use_dist', action='store_true', help='Learn patch distances.')
    parser.add_argument('--use_angles', dest='use_angle', action='store_true', help='Learn patch angles.')
    parser.add_argument('--use_abs_positions', dest='use_abs_positions', action='store_true', 
                        help='Learn absolute positions of patches.')
    parser.add_argument('--use_positional_embeddings', dest='use_positional_embeddings', action='store_true', 
                        help='Use positional embeddings in the ViT.')
    parser.add_argument('--mega_patches', dest='mega_patches', action='store_true', 
                        help='Use megapatches.')
    parser.add_argument('--side_megapatches', type=int, default=5, 
                        help='Output side of images when using the mega-patches.')
    # model params setting
    parser.add_argument('--freeze_fet_enc', dest='freeze_fet_enc', action='store_true', 
                        help='Freeze backbone weights.')
    parser.add_argument('--dropout', type=float, default=0.1, help='The dropout rate of the model.')
    parser.add_argument('--dropout_rel', type=float, default=0.0, 
                        help='The dropout rate applied to the spatial relation classification.')
    parser.add_argument('--smooth_rel', type=float, default=0.0, 
                        help='The label smoothing applied to the spatial relation classification.')
    parser.add_argument('--attn_drop', type=float, default=0.0, 
                        help='Dropout int he attention mask of the model.')
    parser.add_argument('--attn_mask', type=str, default='', 
                        help='Attention mask of the model. Can be empty for standard attention, "diagonal" and "diagonal_clf_row"')
    parser.add_argument('--add_clf_to_patches', dest='add_clf_to_patches', action='store_true', 
                        help='Add CLF token to the patches representation.')
    parser.add_argument('--drop_path_rate', type=float, default=0.0, help='Dropout path rate fo rht ViT.')
    parser.add_argument('--proj_dropout', type=float, default=0.0, help='Dropout rate of linear proj.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay of the optimizer.')
    parser.add_argument('--use_clf_token', default=True, dest='use_clf_token', action='store_true', help='Use classification token in the backbone')

    # data augmentation params
    parser.add_argument('--patch_trans', type=str, default='colJitter:0.8-grayScale:0.2', help='The list of transformations for patches.')

    # parse args
    args = parser.parse_args()
    num_classes = {
        'cifar10': 10,
        'cifar10_14':10,
        'cifar10_28':10,
        'cifar10_36': 10,
        'cifar10_56': 10,
        'cifar10_112': 10,
        'cifar10_224': 10,
        'cifar100': 100,
        'cifar100_224': 100,
        'tiny_imagenet': 200,
        'flower102': 102,
        'flower102_112':102,
        'flower102_56':102,
        'tiny_imagenet_224': 200,
        'svhn': 10,
        'svhn_224': 10,
        'imagenet100': 100,
        'imagenet100_112': 100,
        'imagenet100_56': 100,
        'imagenet100_28': 100,
        'imagenet_subset':1000,
    }
    if args.labels_path:
        with open(args.labels_path, 'r') as f:
            num_classes['imagenet_subset'] = len(f.readlines())
    args.num_classes = num_classes[args.dataset]
    
    if args.task == 'downstream':
        args.metric_monitor = 'val_acc_clf'
        args.metric_maxmin = 'max'
    else:
        args.metric_monitor = 'val_loss'
        args.metric_maxmin = 'min'

    # path size
    if args.patch_size == -1:
        patch_size = {
            'cifar10': 4,
            'cifar10_14': 2,
            'cifar10_28': 4,
            'cifar10_36': 4,
            'cifar10_56': 8,
            'cifar10_112': 16,
            'cifar10_224': 32,
            'cifar100': 4,
            'cifar100_224': 32,
            'tiny_imagenet': 8,
            'tiny_imagenet_224': 32,
            'flower102': 32,
            'flower102_112': 16,
            'flower102_56': 8,
            'svhn': 4,
            'svhn_224': 32,
            'imagenet100': 32,
            'imagenet100_112': 16,
            'imagenet100_56': 8,
            'imagenet100_28': 4,
            'imagenet_subset': 32,
        }
        args.patch_size = patch_size[args.dataset]
    img_height = {
        'cifar10': 32,
        'cifar10_14': 14,
        'cifar10_28': 28,
        'cifar10_36': 36,
        'cifar10_56': 56,
        'cifar10_112': 112,
        'cifar10_224': 224,
        'cifar100': 32,
        'cifar100_224': 224,
        'tiny_imagenet': 64,
        'tiny_imagenet_224': 224,
        'flower102': 224,
        'flower102_112': 112,
        'flower102_56': 56,
        'svhn_224':224,
        'imagenet100': 224,
        'imagenet100_112': 112,
        'imagenet100_56': 56,
        'imagenet100_28': 28,
        'imagenet_subset': 224,
    }
    args.img_height = img_height[args.dataset]
    
    # use_clf_token
    if not args.use_clf_token:
        if args.backbone != 'vit':
            raise Exception(f'Error. Backbone {args.backbone} working without classification token not supported.')
    if verbose:
        args_dict = vars(args)
        logging.info('Input Args: ' + json.dumps({k: args_dict[k] for k in sorted(args_dict)}, indent=4))
    return args

def get_trainer(args, dls):
    """
    Return the PyTorch Lightning Trainer.
    """

    # Logger and callbacks
    logger = get_logger(args)
    callbacks = get_callbacks(args)

    # Trainer
    trainer_args = {
        'gpus': args.num_gpus,
        'max_epochs': args.epochs,
        'deterministic': True,
        'callbacks': callbacks,
        'logger': logger,
        'gradient_clip_val': args.grad_clip_val,
        'max_steps': args.epochs * len(dls['train_aug'])
    }
    trainer = pl.Trainer(**trainer_args)
    return trainer

def accuracy(preds, labels, preds_with_logits=True):
    """
    Compute the accuracy.

    Args:
        preds: tensor of shape [bs, num_classes(, ...)].
        labels: tensor of shape [bs(, ...)].
        preds_with_logits: bool.

    Output:
        acc: scalar.
    """
    if preds_with_logits:
        preds = F.softmax(preds, 1)
    preds = preds.argmax(1)
    acc = (1.0 * (preds == labels)).mean()
    return acc

def cross_entropy(logits, labels, smooth=0.0, dropout=0.0):
    """
    Compute the cross entropy.

    Args:
        logits: tensor of shape [bs, num_classes(, ...)].
        labels: tensor of shape [bs(, num_classes, ...)].
        smooth: scalar, for label smoothing.

    Output:
        xent: scalar.
    """
    if smooth == 0.0:
        return F.cross_entropy(logits, labels)

    logprobs = F.log_softmax(logits, dim=1)
    nll_loss = - logprobs.gather(dim=1, index=labels.unsqueeze(1))
    nll_loss = nll_loss.squeeze(1)
    smooth_loss = - logprobs.mean(dim=1)
    loss = (1.0 - smooth) * nll_loss + smooth * smooth_loss 
    if dropout > 0.0:
        loss = F.dropout(loss, p=dropout)
    return loss.mean()

def cos_anneal(e0, e1, t0, t1, e):
    """ ramp from (e0, t0) -> (e1, t1) through a cosine schedule based on e \in [e0, e1] """
    alpha = max(0, min(1, (e - e0) / (e1 - e0))) # what fraction of the way through are we
    alpha = 1.0 - math.cos(alpha * math.pi / 2) # warp through cosine
    t = alpha * t1 + (1 - alpha) * t0 # interpolate accordingly
    return t

def linear(e0, e1, t0, t1, e):
    """ linear from (e0, t0) -> (e1, t1) through a cosine schedule based on e \in [e0, e1] """
    alpha = max(0, min(1, (e - e0) / (e1 - e0))) # what fraction of the way through are we
    t = alpha * t1 + (1 - alpha) * t0 # interpolate accordingly
    return t

def cos_anneal_warmup(e0, e1, t0, t1, e_w, e):
    if e >= e_w:
        t = cos_anneal(e0, e1, t0, t1, e)
    else:
        t = linear(e0, e_w, 0, t0, e)
    return t

class DecayLR(pl.Callback):
    def __init__(self, lr_init=3e-4, lr_end=1.25e-6, log_lr=False):
        super(DecayLR, self).__init__()
        self.lr_init = lr_init
        self.lr_end = lr_end
        self.log_lr = log_lr

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        # The step size is annealed from 1e10−4 to 1.25e10−6 over 1,200,000 updates. I use 3e-4
        #t = cos_anneal(0, trainer.max_steps, self.lr_init, self.lr_end, trainer.global_step)
        t = cos_anneal_warmup(0, trainer.max_steps, self.lr_init, self.lr_end, trainer.max_steps // 10, trainer.global_step)
        for g in pl_module.optimizer.param_groups:
            g['lr'] = t

        if self.log_lr:
            pl_module.log('lr', t)

def get_callbacks(args):
    """
    Callbacks for the PyTorch Lightning Trainer.
    """

    # Model checkpoint
    model_checkpoint_clbk = pl.callbacks.model_checkpoint.ModelCheckpoint(
        dirpath=None,
        filename='best',
        monitor=args.metric_monitor,
        save_last=True,
        mode=args.metric_maxmin,
    )
    model_checkpoint_clbk.CHECKPOINT_NAME_LAST = '{epoch}-{step}'
    callbacks = [
        model_checkpoint_clbk,
        DecayLR(lr_init=args.lr),
    ]
    return callbacks

def get_logger(args):
    """
    Logger for the PyTorchLightning Trainer.
    """
    logger_kind = 'tensorboard' if 'logger' not in args.__dict__ else args.logger
    if logger_kind == 'tensorboard':
        logger = pl.loggers.tensorboard.TensorBoardLogger(
            save_dir=os.path.join(os.getcwd(), 'tmp'),
            name=args.dataset,
        )

    elif logger_kind == 'wandb':
        task_str = []
        if args.use_supervision:
            task_str += ['clf']
        if args.use_relations:
            task_str += ['rel']
        if args.use_dist:
            task_str += ['dist']
        if args.use_angle:
            task_str += ['angle']
        if args.use_abs_positions:
            task_str += ['abs_pos']
        name = [
            str(args.exp_id), args.dataset, '-'.join(task_str), 
            '-'.join([str(args.backbone), str(args.model_size), str(args.patch_size), str(args.img_height)])
        ]
        if len(args.attn_mask):
            name += [f'attn_mask-{args.attn_mask}']
        if args.add_clf_to_patches:
            name += ['add_clf_to_patches']
        if args.mega_patches:
            name += [f'mega_patches_{args.side_megapatches}']
        logger = pl.loggers.WandbLogger(
            save_dir=os.path.join(os.getcwd(), 'tmp'),
            name='/'.join(name),
            project='relvit_v2',
            settings=wandb.Settings(start_method="fork"),
            config=args,
        )

    else:
        raise Exception(f'Error. Logger "{lokker_kind}" is not supported.')
    return logger

def get_pos_embedding_sim(pos_embed):
    """
    inputs:
        pos_embed: [1, num_embeddings, embed_dim]
    outputs:
        sim_matrix: [num_patches, num_patches, num_patches, num_patches]
    """
    pos_embed = pos_embed.detach()[0, 1:] # don't consider the clf token
    num_embed = pos_embed.shape[0]
    num_patches = int(np.sqrt(num_embed))
    sim_matrix = np.zeros((num_patches, num_patches, num_patches, num_patches))

    for i in range(num_patches):
        for j in range(num_patches):
            ij_flat = i * num_patches + j
            p_ij = pos_embed[ij_flat]
            for k in range(num_patches):
                for l in range(num_patches):
                    kl_flat = k * num_patches + l
                    p_kl = pos_embed[kl_flat]
                    sim_matrix[i, j, k, l] = F.cosine_similarity(p_ij, p_kl, dim=0)
    return torch.tensor(sim_matrix)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def get_split_idxs(ds, train_size=0.8):
    """
    Split a torch dataset into train and validation and return the train and validation idxs.
    """
    from sklearn import model_selection
    from tqdm import tqdm
    SEED = 35571
    cls = []
    for _, y in tqdm(ds):
        cls += [y]
    idx_train, idx_val = model_selection.train_test_split(np.arange(len(ds)), 
                                                          random_state=SEED, 
                                                          shuffle=True, train_size=train_size, 
                                                          stratify=cls)
    return sorted(idx_train.tolist()), sorted(idx_val.tolist())

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def download_file(url, fname):
    folder = os.path.dirname(fname)
    if not os.path.exists(folder) and (len(folder) > 0):
        os.makedirs(folder)
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
        desc=fname,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def extract_tar(tarpath, dest=None):
    """
    Extract a tar file (".tar", ".tar.gz") in dest. If dest is None the tar
    will be extracted in the same folder of the tarpath.
    """
    if dest is None:
        folder = os.path.dirname(tarpath)
        dest = folder if len(folder) > 0 else './'
    tar = tarfile.open(tarpath)
    tar.extractall(dest)
    tar.close()
    logging.info(f'Extracted tar to {folder}')

def get_dataset_constant(ds):
    """
    Given a pytorch dataset of images, compute the normalization tensors (mean and std) used for transforming 
    the input images.
    """
    mu, energy = torch.zeros((3,)), torch.zeros((3,))
    num_samples = len(ds)
    for x, _ in tqdm(ds):
        mu += x.mean(-1).mean(-1) / num_samples
        energy += x.pow(2).mean(-1).mean(-1) / num_samples
    return mu, torch.sqrt(energy - mu.pow(2))
