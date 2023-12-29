import torch
import torch.nn.functional as Func
from helper_functions import *


def get_mask_loss(F_1, F_2, mask):
    """ Returns the mask loss.

    # Parameters:
        @F_1: torch.tensor size [1, C, H, W], the feature map of the first image
        @F_2: torch.tensor size [1, C, H, W], the feature map of the second image
        @mask: torch.tensor size [H, W], the segmentation mask.
            NOTE: 1 encodes what areas should move and 0 what areas should stay fixed.

    # Returns: torch.tensor of size [1], the mask loss
    """
    # TODO: 4. Calculate mask loss
    loss = torch.mean(torch.abs((F_2 - F_1) * (1 - mask)))
    return loss


def get_neighbourhood(p, radius):
    """ Returns a neighbourhood of points around p.

    # Parameters:
        @p: torch.tensor size [2], the current handle point p
        @radius: int, the radius of the neighbourhood to return

    # Returns: torch.tensor size [radius * radius, 2], the neighbourhood of points around p, including p
    """
    # TODO: 1. Get Neighbourhood
    # Note that the order of the points in the neighbourhood does not matter.
    # Do not use for-loops, make use of Pytorch functionalities.
    # neighbor = torch.zeros((2 * radius + 1, 2), device=p.device)
    x = torch.linspace(p[0] - radius, p[0] + radius, 2 * radius + 1, device=p.device)
    y = torch.linspace(p[1] - radius, p[1] + radius, 2 * radius + 1, device=p.device)
    neighbor = torch.meshgrid(x, y)
    neighbor = torch.stack(neighbor, dim=-1).view(-1, 2)
    return neighbor  # Initialize placeholder such that the code runs


def sample_p_from_feature_map(q_N, F_i):
    """ Samples the feature map F_i at the points q_N.

    # Parameters:
        @q_N: torch.tensor size [N, 2], the points to sample from the feature map
        @F_i: torch.tensor size [1, C, H, W], the feature map of the current image

    # Returns: torch.tensor size [N, C], the sampled features at q_N
    """
    assert F_i.shape[-1] == F_i.shape[-2]

    # TODO: 2. Sample features from neighbourhood
    # NOTE: As the points in q_N are floats, we can not access the points from the feature map via indexing.
    # Bilinear interpolation is needed, PyTorch has a function for this: F.grid_sample.
    # NOTE: To check whether you are using grid_sample correctly, you can pass an index matrix as the feature map F_i
    # where each entry corresponds to its x,y index. If you sample from this feature map, you should get the same points back.
    n, c, h, w = F_i.shape
    q_N = torch.clamp(q_N, min=0)
    q_N[:, 0] = torch.clamp(q_N[:, 0], max=w - 1)
    q_N[:, 1] = torch.clamp(q_N[:, 1], max=h - 1)

    q_N[:, 0] = 2 * q_N[:, 0].clone() / max(w - 1, 1) - 1.0
    q_N[:, 1] = 2 * q_N[:, 1].clone() / max(h - 1, 1) - 1.0
    q_N = q_N.flip(-1)
    grid = q_N.unsqueeze(0).unsqueeze(0)
    output = Func.grid_sample(F_i, grid, mode='bilinear', align_corners=False)
    output = output.squeeze(0, 2).transpose(0, 1)

    return output  # Initialize placeholder such that the code runs


def nearest_neighbour_search(f_p, F_q_N, q_N):
    """ Does a nearest neighbourhood search in feature space to find the new handle point position.

    # Parameters:
        @f_p: torch.tensor size [1, C], the feature vector of the handle point p
        @F_q_N: torch.tensor size [N, C], the feature vectors of the neighbourhood points
        @q_N: torch.tensor size [N, 2], the neighbourhood points corresponding to the feature vectors in F_q_N

    # Returns: torch.tensor size [2], the new handle point p
    """
    # TODO: 3. Neighbourhood search
    p = q_N[torch.argmin(torch.norm(F_q_N - f_p, dim=-1))]
    return p  # Initialize placeholder such that the code runs
