"""
--------------------------------------------------------------------------------
TO DO
--------------------------------------------------------------------------------
    - make experiment with learning center vs others 8 locations.
    - make experiment original absolute location.
    - make experiment with some disance measures (manhattan) or position.
    - make experiment with reconstruction of masked input.

--------------------------------------------------------------------------------
SPATIAL RELATIONS AND THEIR ENCODING
--------------------------------------------------------------------------------
GOAL:
    - Create the connectivity matrix of the patches. For each couple of patch 
    there are two spatial relations that are symmetrical:
    Example:
        patch_1 | "is on the left of" | patch_2
        patch_2 | "is the right of"   | patch_1

We identify nine spatial relations:
- LU: left-up
- CU: center-up
- RU: right-up
- LC: left-center
- CC: center-center
- RC: right-center
- LD: left-down
- CD: center-down
- RD: right-down

and we encode these nine classes into (cl_x, cl_y) classes. More specifically:
- cl_x: can vary in {L, C, R}
- cl_y: can vary in {U, C, D}


--------------------------------------------------------------------------------
EXAMPLES:
--------------------------------------------------------------------------------
- 3x3 spatial relations from the center patch (C):
    
    | LU | CU | RU | 
    | LC | CC | RC |
    | LD | CD | RD |

- 5x5 spatial relations from the center patch (C):
    
    | LU | LU | CU | RU | RU | 
    | LU | LU | CU | RU | RU |
    | LC | LC | CC | RC | RC |
    | LD | LD | CD | RD | RD |
    | LD | LD | CD | RD | RD |


--------------------------------------------------------------------------------
CONNECTIVITY MATRIX:
--------------------------------------------------------------------------------
The connectivity matrix is of size (num_patches, num_patches) because for each 
couple of patches there is a spatial relation (and its symmetrical).
"""


##################################################
# Imports
##################################################

import math
import torch
import numpy as np


# utils
def labels2str(x):
    """

    """
    labels_str = ['LU', 'CU', 'RU', 'LC', 'CC', 'RC', 'LD', 'CD', 'RD']
    if isinstance(x, int):
        x = labels_str[x]
    elif isinstance(x, list) or isinstance(x, tuple):
        x = [labels_str[x_i] for x_i in x]
    elif isinstance(x, np.ndarray):
        shape = x.shape
        x = [labels_str[x_i] for x_i in x.reshape(-1)]
        x = np.array(x).reshape(shape)
    elif torch.is_tensor(x):
        x = x.detach().cpu().numpy()
        shape = x.shape
        x = [labels_str[x_i] for x_i in x.reshape(-1)]
        x = np.array(x).reshape(shape)
    else:
        raise Exception(f'Error. The input type of x "{type(x)}" is not supported.')
    return x

def labels2int(x):
    """

    """
    labels_str = ['LU', 'CU', 'RU', 'LC', 'CC', 'RC', 'LD', 'CD', 'RD']
    if isinstance(x, int):
        x = labels_str.index(x)
    elif isinstance(x, list) or isinstance(x, tuple):
        x = [labels_str.index(x_i) for x_i in x]
    elif isinstance(x, np.ndarray):
        shape = x.shape
        x = [labels_str.index(x_i) for x_i in x.reshape(-1)]
        x = np.array(x).reshape(shape)
    elif torch.is_tensor(x):
        x = x.detach().cpu().numpy()
        shape = x.shape
        x = [labels_str.index(x_i) for x_i in x.reshape(-1)]
        x = np.array(x).reshape(shape)
    else:
        raise Exception(f'Error. The input type of x "{type(x)}" is not supported.')
    return x

# create spatial classes
def create_label_matrix(num_patches):
    """
    Create the spatial location matrix (as the example above).
    """
    labels = np.arange(9) # Spatial location encoded as int...
    labels_matrix = labels.reshape(3, 3)
    patch_matrix_side = int(math.sqrt(num_patches))

    side = 2 * patch_matrix_side - 1
    lu, cu, ru = labels_matrix[0, 0], labels_matrix[0, 1], labels_matrix[0, 2]
    lc, cc, rc = labels_matrix[1, 0], labels_matrix[1, 1], labels_matrix[1, 2]
    ld, cd, rd = labels_matrix[2, 0], labels_matrix[2, 1], labels_matrix[2, 2]

    corner_side = int((side - 1) / 2)
    cor = np.ones([corner_side, corner_side]).astype(labels.dtype)
    cen = np.ones([1, 1]).astype(labels.dtype)
    ver = np.ones([corner_side, 1]).astype(labels.dtype)
    hor = np.ones([1, corner_side]).astype(labels.dtype)

    labels_matrix = np.concatenate([
        np.concatenate([lu * cor, cu * ver, ru * cor], 1),
        np.concatenate([lc * hor, cc * cen, rc * hor], 1),
        np.concatenate([ld * cor, cd * ver, rd * cor], 1),
    ], 0)

    return labels_matrix

def create_patch_matrix(num_patches):
    side = int(math.sqrt(num_patches))
    patches = np.arange(num_patches).reshape(side, side)
    return patches

def create_connectivity_matrix(patches):
    patch_flat = patches.reshape(-1)
    num_patches = len(patch_flat)
    labels_matrix = create_label_matrix(num_patches)

    side = int(math.sqrt(num_patches))
    conn_matrix = - 1 * np.ones([num_patches, num_patches]).astype(labels_matrix.dtype)
    center = int((labels_matrix.shape[0] - 1) / 2)
    for i in range(side):
        for j in range(side):
            labels_ij_matrix = labels_matrix[center - i: center - i  + side, center - j: center - j  + side]
            labels_ij_matrix = labels_ij_matrix.reshape(-1)
            #idx_flat = patch_flat[i * side + j]
            idx_flat = i * side + j
            conn_matrix[idx_flat, :] = labels_ij_matrix

    conn_matrix = conn_matrix.T
    conn_matrix = conn_matrix[patch_flat, :][:, patch_flat]
    return conn_matrix

def create_dist_matrix(patches, norm=2, vmin=-1, vmax=1):
    patch_flat = patches.reshape(-1)
    num_patches = len(patch_flat)
    side = int(np.sqrt(num_patches))
    dist_matrix = np.zeros((num_patches, num_patches), dtype=np.float32)
    x = torch.arange(side).unsqueeze(0).repeat(side, 1)
    y = torch.arange(side).unsqueeze(1).repeat(1, side)
    p = torch.stack([x, y], -1).to(torch.float32).view(-1, 2)
    dist_matrix = torch.cdist(p.unsqueeze(0), p.unsqueeze(0), p=norm).squeeze(0)
    dist_matrix = dist_matrix[patch_flat, :][:, patch_flat]
    dist_matrix = dist_matrix / dist_matrix.max() * (vmax - vmin) + vmin # scale in range [vmin, vmax]
    return dist_matrix

def create_angle_matrix(patches, vmin=-1, vmax=1):
    patch_flat = patches.reshape(-1)
    num_patches = len(patch_flat)
    side = int(np.sqrt(num_patches))
    dist_matrix = np.zeros((num_patches, num_patches), dtype=np.float32)
    x = torch.arange(side).unsqueeze(0).repeat(side, 1) + side / 2
    y = torch.arange(side).unsqueeze(1).repeat(1, side) + side / 2
    p = torch.stack([x, y], -1).to(torch.float32).view(-1, 2)
    angle_matrix = np.zeros((num_patches, num_patches), dtype=np.float32)
    for i, p_i in enumerate(p):
        for j, p_j in enumerate(p):
            if i == j:
                angle_matrix[i, j] = 0.0
            else:
                denom = torch.norm(p_i) * torch.norm(p_j)
                num = (p_i * p_j).sum()
                angle_matrix[i, j] = torch.acos(num / (denom + 1e-4)) / np.pi
    angle_matrix = angle_matrix[patch_flat, :][:, patch_flat]
    angle_matrix = angle_matrix / angle_matrix.max() * (vmax - vmin) + vmin # scale in range [vmin, vmax]
    return angle_matrix

def create_abs_positions_matrix(patches):
    patch_flat = patches.reshape(-1)
    num_patches = len(patch_flat)
    abs_positions_matrix = torch.arange(num_patches)
    return abs_positions_matrix


class SSLSignal():
    def __init__(self, use_relations=True, use_distances=True, use_angles=True, use_abs_positions=True):
        patches = None #torch.arange(side * side).view(side, side)
        self.use_relations = use_relations
        self.use_distances = use_distances
        self.use_angles = use_angles
        self.use_abs_positions = use_abs_positions
        self.matrices_initialized = False
        if self.use_relations:
            self.conn_matrix = None
        if self.use_distances:
            self.dist_matrix = None
        if self.use_angles:
            self.angle_matrix = None
        if self.use_abs_positions:
            self.abs_position_matrix = None

    def __call__(self, patches):
        patch_flat = patches.reshape(-1)
        if not self.matrices_initialized:
            num_patches = patch_flat.shape[0]
            side = int(np.sqrt(num_patches))
            patches_init = torch.arange(num_patches).reshape(side, side).numpy()
            if self.use_relations:
                self.conn_matrix = create_connectivity_matrix(patches_init)
                self.conn_matrix = torch.from_numpy(self.conn_matrix).to(torch.int64)
            if self.use_distances:
                self.dist_matrix = create_dist_matrix(patches_init)
            if self.use_angles:
                self.angle_matrix = create_angle_matrix(patches_init)
            if self.use_abs_positions:
                self.abs_positions_matrix = create_abs_positions_matrix(patches_init)
            self.matrices_initialized = True
        out = {}
        if self.use_relations:
            out['relations'] = self.conn_matrix[patch_flat, :][:, patch_flat]
        if self.use_distances:
            out['distances'] = self.dist_matrix[patch_flat, :][:, patch_flat]
        if self.use_angles:
            out['angles'] = self.angle_matrix[patch_flat, :][:, patch_flat]
        if self.use_abs_positions:
            out['abs_positions'] = self.abs_positions_matrix[patch_flat]
        return out
