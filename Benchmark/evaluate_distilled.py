import numpy as np
from scipy import linalg

import torch
import torch.nn.functional as F

from torchvision.datasets import CIFAR10
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.models.feature_extraction import create_feature_extractor


def load_distilled(filepath, device='cpu'):
    """
    Expects .pth with keys:
      'X': [Tensor(n, C, H, W), ...]   # list of synthetic stages
      'Y': [Tensor(n), ...]
    Returns list of (X_stage, Y_stage) tensors on `device`.
    """
    data = torch.load(filepath, map_location='cpu')
    X_list = [x.to(device) for x in data['X']]
    Y_list = [y.to(device) for y in data['Y']]
    return list(zip(X_list, Y_list))


def get_real_mnist_loader(batch_size=64, num_workers=0, real_subset_size=10000):
    tf = transforms.Compose([
        transforms.Resize(299),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
    ds = datasets.MNIST(root='data', train=True, download=True, transform=tf)
    if real_subset_size is not None:
        # pick real_subset_size random indices out of 60_000
        indices = torch.randperm(len(ds))[:real_subset_size].tolist()
        sampler = SubsetRandomSampler(indices)
        return DataLoader(ds, batch_size=batch_size,
                          sampler=sampler, num_workers=num_workers,
                          pin_memory=True)
    else:
        # use the full dataset
        return DataLoader(ds, batch_size=batch_size,
                          shuffle=True, num_workers=num_workers,
                          pin_memory=True)

# a reusable normalizer for 3-channel tensors
norm3 = transforms.Normalize(mean=[0.5,0.5,0.5],
                             std =[0.5,0.5,0.5])


def get_real_cifar10_loader(batch_size=64, num_workers=0, real_subset_size=10000):
    tf = transforms.Compose([
        transforms.Resize(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
    ds = CIFAR10(root='data', train=True, download=True, transform=tf)
    if real_subset_size is not None:
        # pick real_subset_size random indices out of 60_000
        indices = torch.randperm(len(ds))[:real_subset_size].tolist()
        sampler = SubsetRandomSampler(indices)
        return DataLoader(ds, batch_size=batch_size,
                          sampler=sampler, num_workers=num_workers,
                          pin_memory=True)
    else:
        # use the full dataset
        return DataLoader(ds, batch_size=batch_size,
                          shuffle=True, num_workers=num_workers,
                          pin_memory=True)

# a reusable normalizer for 3-channel tensors
norm3 = transforms.Normalize(mean=[0.5,0.5,0.5],
                             std =[0.5,0.5,0.5])


def get_inception_feature_extractor(device='cpu'):
    from torchvision.models import inception_v3, Inception_V3_Weights
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT,
                         transform_input=False).to(device)
    # disable aux classifier
    model.aux_logits = False
    model.eval()
    # extract the pooled 2048-d vector before the final fc
    return create_feature_extractor(model, return_nodes={'avgpool': 'feat'}).to(device)


@torch.no_grad()
def compute_activations(dataloader, feat_extractor, device='cpu'):
    feats = []
    for imgs, _ in dataloader:
        imgs = imgs.to(device)
        # — if grayscale 1×28×28, turn into 3×299×299
        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        if imgs.shape[2] != 299 or imgs.shape[3] != 299:
            imgs = F.interpolate(imgs,
                                 size=(299, 299),
                                 mode='bilinear',
                                 align_corners=False)
        # normalize exactly as real loader does
        imgs = norm3(imgs)

        out = feat_extractor(imgs)['feat']     # [B, 2048, 1, 1]
        out = out.reshape(out.size(0), -1)     # → [B, 2048]
        feats.append(out.cpu().numpy())
    return np.concatenate(feats, axis=0)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet distance."""
    diff = mu1 - mu2
    # sqrt of product of covariances
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1+offset).dot(sigma2+offset))
    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)


def compute_statistics(feats):
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def gaussian_kernel(x, y, sigma):
    """Compute Gaussian (RBF) kernel matrix between x and y."""
    x_norm = (x**2).sum(1).reshape(-1, 1)
    y_norm = (y**2).sum(1).reshape(1, -1)
    dist2 = x_norm + y_norm - 2 * x.dot(y.T)
    return np.exp(-dist2 / (2 * sigma**2))


def calculate_mmd(feats1, feats2, sigma=None):
    """Unbiased estimate of squared MMD."""
    if sigma is None:
        # median heuristic on a small subset
        all_feats = np.vstack([feats1[:500], feats2[:500]])
        dists = np.sqrt(
            ((all_feats[:, None] - all_feats[None, :])**2).sum(-1)
        )
        sigma = np.median(dists)
    K_xx = gaussian_kernel(feats1, feats1, sigma)
    K_yy = gaussian_kernel(feats2, feats2, sigma)
    K_xy = gaussian_kernel(feats1, feats2, sigma)

    n, m = feats1.shape[0], feats2.shape[0]
    # unbiased estimators
    sum_xx = (K_xx.sum() - np.trace(K_xx)) / (n * (n - 1))
    sum_yy = (K_yy.sum() - np.trace(K_yy)) / (m * (m - 1))
    sum_xy = K_xy.sum() / (n * m)
    return sum_xx + sum_yy - 2 * sum_xy


def evaluate_distillation(distilled_path,
                           dataset='mnist',
                           device='cuda' if torch.cuda.is_available() else 'cpu',
                           batch_size=64):
    # load real and distilled
    # pick real loader based on dataset
    if dataset == 'mnist':
        real_loader = get_real_mnist_loader(batch_size=batch_size, num_workers=0)
    else:  # cifar10
        real_loader = get_real_cifar10_loader(batch_size=batch_size, num_workers=0)
    distilled = load_distilled(distilled_path, device)

    # prepare feature extractor
    feat_extractor = get_inception_feature_extractor(device)

    # compute real activations once
    print("→ Computing activations for real MNIST…")
    real_feats = compute_activations(real_loader, feat_extractor, device)
    mu_real, sigma_real = compute_statistics(real_feats)

    # evaluate each stage
    results = []
    for stage_idx, (X_syn, _) in enumerate(distilled, 1):
        # build a dataloader over synthetic batch
        ds_syn = torch.utils.data.TensorDataset(X_syn,
                      torch.zeros(len(X_syn), dtype=torch.long))
        loader_syn = DataLoader(ds_syn, batch_size=batch_size, shuffle=False)

        print(f"→ Stage {stage_idx}: computing activations for synthetic…")
        syn_feats = compute_activations(loader_syn, feat_extractor, device)
        mu_syn, sigma_syn = compute_statistics(syn_feats)

        fid = calculate_frechet_distance(mu_real, sigma_real, mu_syn, sigma_syn)
        mmd = calculate_mmd(real_feats, syn_feats)

        results.append({'stage': stage_idx, 'FID': float(fid), 'MMD': float(mmd)})
        print(f"   • Stage {stage_idx} → FID: {fid:.2f}, MMD: {mmd:.4f}")

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--distilled', type=str,
                        default=None,
                        help='Path to distilled .pth (if omitted, uses data/Distilled/meta-model-matching_<dataset>_convnet.pt)')
    parser.add_argument('--dataset', choices=['mnist','cifar10'],
                        default='mnist',
                        help='Which real dataset to compare against')
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()


    # fill in default distilled path if needed
    if args.distilled is None:
        args.distilled = f'data/Distilled/meta-model-matching_{args.dataset}_convnet.pt'


    res = evaluate_distillation(
                distilled_path=args.distilled,
                dataset=args.dataset,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                batch_size=args.batch_size)

    print("\n=== Summary ===")
    for r in res:
        print(f"Stage {r['stage']}:  FID = {r['FID']:.2f},  MMD = {r['MMD']:.4f}")