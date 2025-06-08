# augment.py
import torch
import torch.nn.functional as F


class DiffAug:
    """
    Differentiable augmentation pipeline for MNIST/CIFAR10.
    Usage:
        aug = DiffAug(dataset='mnist')
        x_aug = aug(x)   # preserves gradient graph
    """

    def __init__(
        self,
        dataset: str = 'mnist',
        p_flip: float = 0.2,
        max_rot: float = 15.0,          # degrees
        max_trans: float = 0.125,       # fraction of H/W
        brightness: float = 0.2,        # ± of pixel range
        contrast: float = 0.2,          # ± relative to mean
        cutout_ratio: float = 0       # fraction of H/W
    ):
        self.dataset     = dataset.lower()
        self.p_flip      = p_flip
        self.max_rot     = max_rot * torch.pi / 180.0  # to radians
        self.max_trans   = max_trans
        self.brightness  = brightness
        self.contrast    = contrast
        self.cutout_ratio= cutout_ratio

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W), in [0,1]
        x = self._flip(x)
        x = self._rotate(x)
        x = self._translate(x)
        x = self._jitter_brightness(x)
        x = self._jitter_contrast(x)
        x = self._cutout(x)
        return x

    def _flip(self, x):
        if torch.rand(1, device=x.device) < self.p_flip:
            return x.flip(-1)
        return x

    def _rotate(self, x):
        B, C, H, W = x.shape
        angles = (torch.rand(B, device=x.device) * 2 - 1) * self.max_rot
        cos = angles.cos()
        sin = angles.sin()
        # build batch affine matrices
        theta = torch.zeros(B, 2, 3, device=x.device)
        theta[:,0,0] =  cos
        theta[:,0,1] = -sin
        theta[:,1,0] =  sin
        theta[:,1,1] =  cos
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        return F.grid_sample(x, grid, padding_mode='border', align_corners=False)

    def _translate(self, x):
        B, C, H, W = x.shape
        # translation as fraction of H/W
        max_t = self.max_trans
        tx = (torch.rand(B, device=x.device) * 2 - 1) * max_t
        ty = (torch.rand(B, device=x.device) * 2 - 1) * max_t
        theta = torch.zeros(B, 2, 3, device=x.device)
        theta[:,0,0] = 1
        theta[:,1,1] = 1
        theta[:,0,2] = tx
        theta[:,1,2] = ty
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        return F.grid_sample(x, grid, padding_mode='border', align_corners=False)

    def _jitter_brightness(self, x):
        # add uniform noise in [-brightness, +brightness]
        B = x.shape[0]
        b = (torch.rand(B, 1, 1, 1, device=x.device) * 2 - 1) * self.brightness
        return x + b

    def _jitter_contrast(self, x):
        # scale (x - mean)*c + mean
        B, C, H, W = x.shape
        c = torch.rand(B, 1, 1, 1, device=x.device) * self.contrast + (1 - self.contrast/2)
        mean = x.mean(dim=[2,3], keepdim=True)
        return (x - mean) * c + mean

    def _cutout(self, x):
        B, C, H, W = x.shape
        cut_h = int(H * self.cutout_ratio)
        cut_w = int(W * self.cutout_ratio)

        # random centers
        cx = torch.randint(0, H, (B,), device=x.device)
        cy = torch.randint(0, W, (B,), device=x.device)

        # build masks via broadcasting
        grid_x = torch.arange(H, device=x.device).view(1, H, 1)  # (1,H,1)
        grid_y = torch.arange(W, device=x.device).view(1, 1, W)  # (1,1,W)
        # (B,H,W)
        mask = (( (grid_x - cx.view(B,1,1)).abs() > (cut_h//2) ) |
                ( (grid_y - cy.view(B,1,1)).abs() > (cut_w//2) ))
        mask = mask.unsqueeze(1).float()  # (B,1,H,W)
        return x * mask


# end of augment.py
