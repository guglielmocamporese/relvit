##################################################
# Imports
##################################################

import math
import logging
import numpy as np
from PIL import Image
from matplotlib import cm
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam, AdamW
from torchvision.utils import make_grid
import pytorch_lightning as pl
import wandb

# Custom
from models.backbones import vit, t2t, swin
from models.layers import MultiHeadAttentionScores
import utils
from utils import accuracy, cross_entropy

cmap = cm.get_cmap('viridis')


##################################################
# RelViT Model
##################################################

class RelViT(pl.LightningModule):
    """
    Lightning module that defines the Relational Vision Transformer Model.
    """
    def __init__(self, args, num_classes, num_rel_classes=9, use_relations=False, use_positional_emb=True, 
                 use_supervision=True, patch_size=32, dropout=0.1, freeze_fet_enc=False, use_distances=False, 
                 use_angles=False, use_abs_positions=False, proj_dropout=0.0, use_clf_token=True):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self.use_supervision = use_supervision
        self.use_relations = use_relations
        self.use_distances = use_distances
        self.use_angles = use_angles
        self.use_abs_positions = use_abs_positions
        self.use_positional_emb = use_positional_emb
        self.patch_size = patch_size
        self.dropout = dropout
        self.proj_dropout = proj_dropout
        self.freeze_fet_enc = freeze_fet_enc
        self.use_clf_token = use_clf_token
        self.img_height = self.args.img_height if not self.args.mega_patches else self.args.side_megapatches * self.patch_size
        self.num_patches = (self.img_height // self.patch_size) ** 2
        self.backbone = self._get_backbone()
        if self.use_relations:
            self.rel_head = MultiHeadAttentionScores(self.backbone.embed_dim, num_rel_classes)
        if self.use_supervision:
            if self.use_clf_token:
                self.clf_head = nn.Linear(self.backbone.embed_dim, args.num_classes)
            else:
                self.clf_head = nn.Linear(self.backbone.embed_dim * self.num_patches, args.num_classes)
        if self.use_distances:
            self.dist_head = MultiHeadAttentionScores(self.backbone.embed_dim, 1)
        if self.use_angles:
            self.angle_head = MultiHeadAttentionScores(self.backbone.embed_dim, 1)
        if self.use_abs_positions:
            self.abs_pos_head = nn.Linear(self.backbone.embed_dim, self.num_patches)

        if self.args.backbone == 't2t_vit':
            self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

        if self.freeze_fet_enc:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def _get_backbone(self):
        if self.args.backbone in ['vit']:
            backbone_args = {
                'img_size': self.args.img_height,
                'patch_size': self.args.patch_size,
                'attn_drop_rate': self.args.attn_drop,
                'use_positional_embeddings': self.use_positional_emb,
                'drop_path_rate': self.args.drop_path_rate,
                'proj_dropout': self.proj_dropout,
                'drop_rate': self.dropout,
                'use_clf_token':self.use_clf_token,
                'task':self.args.task, 
                'side_megapatches':self.args.side_megapatches,
            }
            backbone = vit.__dict__[f'vit_{self.args.model_size}'](**backbone_args)
        elif self.args.backbone == 't2t_vit':
            backbone_args = {
                'img_size': self.args.img_height,
                'num_classes': self.args.num_classes,
            }
            backbone = t2t.__dict__[f't2t_vit_{self.args.model_size}'](**backbone_args)
        elif self.args.backbone == "swin":
            backbone_args = {
                'num_classes': self.args.num_classes,
            }
            backbone = swin.__dict__[f'swin_{self.args.model_size}'](**backbone_args)
        return backbone

    def forward(self, x):
        """
        Forward function of the model.
        """
        out = {}
        mask = self.args.attn_mask if len(self.args.attn_mask) > 0 else None
        if self.args.backbone == "swin":
            z_patches = self.backbone.forward_features(x)
        elif self.args.backbone in ['vit', 't2t_vit']:
            if self.args.backbone == 'vit':
                feats = self.backbone(x, mask=mask)
            else:
                feats = self.backbone.forward_features(x)
            if self.args.add_clf_to_patches:
                z_patches = feats['z_patches'] + feats['z'].unsqueeze(1)
            else:
                z_patches = feats['z_patches']  # [B, N_PATCH, D]
                if self.args.backbone == 't2t_vit':
                    B,P,C = z_patches.shape
                    side = int(math.sqrt(P))
                    z_patches = torch.reshape(z_patches.transpose(1,2), [B,C,side,side])
                    z_patches = self.avg_pool(z_patches).flatten(2).transpose(1,2)
        
        if self.use_relations:
            out['rel'] = self.rel_head(z_patches)
        if self.use_distances:
            out['dist'] = self.dist_head(z_patches)
        if self.use_angles:
            out['angle'] = self.angle_head(z_patches)
        if self.use_abs_positions:
            out['abs_pos'] = self.abs_pos_head(z_patches).transpose(1, 2)
        if self.use_supervision:
            if self.args.backbone in ["vit", "t2t_vit"]:
                if self.use_clf_token:
                    out['clf'] = self.clf_head(feats['z'])
                else:
                    out['clf'] = self.clf_head(z_patches.flatten(1))
            else:
                z = z_patches
                z = self.backbone.avgpool(z.transpose(1, 2))  # B C 1
                z = torch.flatten(z, 1)
                out['clf'] = self.backbone.head(z)
        return out

    def training_step(self, batch, batch_idx, part='train'):
        """
        Define a single training step.
        """

        # Retrieve data
        x = batch['images'] # [B, C, H, W]

        # Forward and loss
        logits = self(x)
        loss = 0.0

        if self.use_supervision:
            y_cls = batch['labels'] # [B]
            loss_clf = F.cross_entropy(logits['clf'], y_cls)
            loss += loss_clf
            acc_cls = accuracy(logits['clf'], y_cls)
            self.log(f'{part}_acc_clf', acc_cls, prog_bar=True)
            self.log(f'{part}_loss_clf', loss_clf)

        if self.use_relations:
            y_rel = batch['relations'] # [B, N_PATCH, N_PATCH]
            loss_rel = cross_entropy(logits['rel'], y_rel, smooth=self.args.smooth_rel, 
                                     dropout=self.args.dropout_rel if part == 'train' else 0.0)
            loss += loss_rel
            acc_rel = accuracy(logits['rel'], y_rel)
            self.log(f'{part}_acc_rel', acc_rel, prog_bar=True)
            self.log(f'{part}_loss_rel', loss_rel)

        if self.use_distances:
            y_dist = batch['distances'].unsqueeze(1) # [B, 1, N_PATCH, N_PATCH]
            loss_dist = F.mse_loss(logits['dist'], y_dist)
            loss += loss_dist
            self.log(f'{part}_loss_dist', loss_dist)

        if self.use_angles:
            y_angle = batch['angles'].unsqueeze(1) # [B, 1, N_PATCH, N_PATCH]
            loss_angle = F.mse_loss(logits['angle'], y_angle)
            loss += loss_angle
            self.log(f'{part}_loss_angle', loss_angle)

        if self.use_abs_positions:
            y_abs_pos = batch['abs_positions'] # [B, N_PATCH]
            loss_abs_pos = F.cross_entropy(logits['abs_pos'], y_abs_pos)
            loss += loss_abs_pos
            acc_abs_pos = accuracy(logits['abs_pos'], y_abs_pos)
            self.log(f'{part}_acc_abs_pos', acc_abs_pos, prog_bar=True)
            self.log(f'{part}_loss_abs_pos', loss_abs_pos)

        # Log
        self.log(f'{part}_loss', loss)
        if part == 'train':
            self.log('lr', self.optimizer.param_groups[0]['lr'])
        elif part == 'val':
            if self.use_positional_emb and batch_idx == 0 and self.args.backbone in ['vit']:
                pos_sim_matrix = utils.get_pos_embedding_sim(self.backbone.pos_embed.detach().cpu()) # [npatch, npatch, npatch, npatch]
                num_patches = pos_sim_matrix.shape[0]
                pos_sim_matrix = pos_sim_matrix.view(-1, num_patches, num_patches).unsqueeze(1) # [npatch * npatch, 1, npatch, npatch]
                pos_sim_matrix = (pos_sim_matrix + 1.0) / 2.0 # pos_sim_matrix in [0, 1]
                pos_sim_matrix = make_grid(pos_sim_matrix, nrow=num_patches, padding=1, pad_value=-1)[0] # [npatch + p, npatch + p]
                mask = torch.where(pos_sim_matrix >= 0, torch.ones_like(pos_sim_matrix), torch.zeros_like(pos_sim_matrix))
                mask = torch.stack([torch.ones_like(mask), torch.ones_like(mask), torch.ones_like(mask), mask], -1)
                pos_sim_matrix = torch.tensor(cmap(pos_sim_matrix)) # [npatch + p, npatch + p, 4]
                pos_sim_matrix = (pos_sim_matrix * mask).permute(2, 0, 1) # [4, npatch + p, npatch + p]
                if self.args.logger == 'wandb':
                    self.logger.experiment.log({'pos_embed_sim': [wandb.Image(pos_sim_matrix)]})

            if batch_idx == 0 and self.args.backbone in ['vit']:
                kernel = list(self.backbone.patch_embed.parameters())[0].detach().cpu().numpy()
                D, C, PS, PS = kernel.shape
                pca = PCA(n_components=28)
                rgb_filt = pca.fit_transform(kernel.reshape(D, -1).T).T.reshape(-1, C, PS, PS)
                rgb_filt = (rgb_filt - rgb_filt.min()) / (rgb_filt.max() - rgb_filt.min())
                rgb_filt = torch.tensor(rgb_filt)
                rgb_filt = make_grid(rgb_filt, nrow=7, padding=1)
                if self.args.logger == 'wandb':
                    self.logger.experiment.log({'rgb_emb_filters': [wandb.Image(rgb_filt)]})


        return loss

    def validation_step(self, batch, batch_idx):
        """
        Define a single validation step.
        """
        return self.training_step(batch, batch_idx, part='val')

    def configure_optimizers(self):
        """
        Configure optimizer for the learning process.
        """
        if self.args.optimizer == 'adam':
            optimizer = Adam(self.parameters(), self.args.lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'sgd':
            optimizer = SGD(self.parameters(), self.args.lr, momentum=0.9, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adamw':
            optimizer = AdamW(self.parameters(), self.args.lr, weight_decay=self.args.weight_decay)
        else:
            raise Exception(f'Error. Optimizer "{self.args.optimizer}" not supported.')
        self.optimizer = optimizer
        return optimizer

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    logging.info(f"Skip loading parameter: {k}, "
                                f"required shape: {model_state_dict[k].shape}, "
                                f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                logging.info(f"Dropping parameter {k}")
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        if self.args.backbone in ["vit", "t2t_vit"]:
            key_load = list(checkpoint.keys())
            if len(key_load) != 1:
                raise Exception(f'Error. checkpoint file extension not supported.')
            state_dict = checkpoint[key_load[0]]
        else:
            state_dict = checkpoint['model']
        model_state_dict = self.backbone.state_dict()

        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    logging.info(f"Skip loading parameter: {k}, "
                                f"required shape: {model_state_dict[k].shape}, "
                                f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                logging.info(f"Dropping parameter {k}")

                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)

        checkpoint['state_dict'] = state_dict
        self.backbone.load_state_dict(checkpoint['state_dict'])

def get_model(args):
    """
    Return the RelViT model.
    """
    model_args = {
        'args': args,
        'num_classes': args.num_classes,
        'num_rel_classes': 9,
        'use_relations': args.use_relations,
        'use_positional_emb': args.use_positional_embeddings,
        'use_supervision': args.use_supervision,
        'use_distances': args.use_dist,
        'use_angles': args.use_angle,
        'patch_size': args.patch_size,
        'dropout': args.dropout,
        'freeze_fet_enc': args.freeze_fet_enc,
        'use_abs_positions': args.use_abs_positions,
        'proj_dropout': args.proj_dropout,
        'use_clf_token':args.use_clf_token,
    }
    model = RelViT(**model_args)
    if len(args.model_checkpoint) > 0:
        if args.model_checkpoint.endswith('.ckpt'):
            model = model.load_from_checkpoint(args.model_checkpoint, strict=False, **model_args)
        else: 
            try:
                model.load_model(args.model_checkpoint)
            except:
                raise Exception(f'Error. checkpoint file extension not supported.')
        logging.info(f'Loaded model at "{args.model_checkpoint}"')
    return model