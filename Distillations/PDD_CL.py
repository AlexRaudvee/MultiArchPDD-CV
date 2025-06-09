import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import SGD, Adam
from tqdm.auto import trange
from torch.func import functional_call
from torch.utils.data import Subset, DataLoader

from Distillations.augment import DiffAug

# torch.autograd.set_detect_anomaly(True)

def sample_class(dataset, class_label, batch_size=64):
    # collect indices whose label == class_label
    indices = [i for i, (_, lbl) in enumerate(dataset) if lbl == class_label]
    subset  = Subset(dataset, indices)
    loader  = DataLoader(subset, batch_size=batch_size, shuffle=True)
    return next(iter(loader))

class PDDCompositeLoss:
    """
    Progressive Dataset Distillation with naive meta-model matching
    (Algorithm 3 from the report), using `higher` for differentiable inner-loop
    and PyTorch optimizers for both student and synthetic updates.

    Args:
        model_fns:       list of callable that returns a fresh nn.Module()
        real_loader:     DataLoader yielding (x_real, y_real)
        image_shape:     tuple (C, H, W)
        num_classes:     number of output classes
        synthetic_size:  number of synthetic examples per stage
        P:               number of PDD stages
        K:               number of synthetic-refinement iterations per stage
        T:               number of fine-tuning steps in inner/outer loops
        lr_model:        learning rate for student inner-loop optimizer
        lr_syn_data:     learning rate for synthetic updates (meta-step)
        syn_optimizer:   'adam' or 'sgd' for updating synthetic variables
        syn_momentum:    momentum for synthetic SGD optimizer
        inner_optimizer: 'sgd', 'momentum' for inner-loop learning
        inner_momentum:  momentum for student SGD with momentum
        inner_betas:     tuple (beta1, beta2) for student Adam optimizer
        inner_eps:       epsilon for student Adam optimizer
        device:          torch.device (defaults to CUDA if available)
    """

    def __init__(
        self,
        model_fns,
        real_loader,
        image_shape,
        num_classes,
        ipc,
        P,
        K,
        T,
        lr_model,
        lr_syn_data,
        regularisation=0.0001,
        warmup_epochs=10,
        composite_weights=[0.5, 0.5],
        syn_optimizer="adam",
        syn_momentum=0.9,
        inner_optimizer="sgd",
        inner_momentum=0.9,
        inner_betas=(0.9, 0.999),
        inner_eps=1e-8,
        debug=False,
        device=None
    ):
        self.model_fns           = model_fns
        self.real_loader        = real_loader
        self.C, self.H, self.W  = image_shape
        self.num_classes        = num_classes
        self.ipc                = ipc
        self.P                  = P
        self.K                  = K
        self.T                  = T
        self.lr_model           = lr_model
        self.lr_syn_data        = lr_syn_data
        self.regularisation     = regularisation
        self.warmup_epochs      = warmup_epochs
        self.composite_weights  = composite_weights
        self.syn_optimizer      = syn_optimizer.lower()
        self.syn_momentum       = syn_momentum
        self.inner_optimizer    = inner_optimizer.lower()
        self.inner_momentum     = inner_momentum
        self.inner_betas        = inner_betas
        self.inner_eps          = inner_eps
        self.device             = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.debug              = debug
        self.smooth_kernel      = 1
        self.m_per_class        = ipc

        # storage for monitoring
        self.synthetic_size = ipc * num_classes
        self.S_X = []
        self.S_Y = []
        self.meta_loss_history = {}  # list of lists per stage
        self.inner_lrs  = nn.Parameter(torch.log(torch.exp(torch.ones(self.T, device=self.device) * lr_model) - 1.0))
        
        # Precompute smoothing kernel
        k = self.smooth_kernel
        kernel = torch.ones((1, 1, k, k), device=self.device) / (k * k)
        self.register_buffer = None  # placeholder
        self.smooth_kernel_tensor = kernel
        
        # Augmentations 
        self.aug       = DiffAug(dataset="mnist", p_flip=0, max_trans=0.05)
        self.aug_rand  = DiffAug(dataset="mnist")

    def _smooth(self, X: torch.Tensor) -> torch.Tensor:
        # Apply average smoothing across each channel independently
        # X: (B, C, H, W)
        pad = self.smooth_kernel_tensor.shape[-1] // 2
        # pad spatial dims
        X_pad = F.pad(X, (pad, pad, pad, pad), mode='reflect')
        # depthwise conv: groups=C applies each kernel to each channel
        smoothed = F.conv2d(X_pad, self.smooth_kernel_tensor, groups=self.C)
        return smoothed
    
    def distill(self):
        # Initial student checkpoint
        dataset = self.real_loader.dataset
        theta_0s = [fn().to(self.device).state_dict() for fn in self.model_fns]
        real_iter = iter(self.real_loader)

        for stage in range(self.P):
            # Initialize new synthetic batch
            X = nn.Parameter(torch.randn(self.ipc * self.num_classes, self.C, self.H, self.W, device=self.device) * 0.05)
            Y = torch.arange(self.num_classes, device=self.device).repeat_interleave(self.ipc)

            # Meta optimizer (only X)
            syn_opt = SGD([X], lr=self.lr_syn_data, momentum=0.9) if False  \
                else Adam([X], lr=self.lr_syn_data)
            stage_losses = []

            # Prepare previous synthetic
            if self.S_X:
                X_prev = torch.cat(self.S_X, dim=0).detach()
                Y_prev = torch.cat(self.S_Y, dim=0).detach()
            else:
                X_prev = torch.empty(0, self.C, self.H, self.W, device=self.device)
                Y_prev = torch.empty(0, dtype=torch.long, device=self.device)

            for k in trange(self.K, desc=f"Stage {stage+1}/{self.P}", leave=False):
                # Build support set
                X_sup = torch.cat([X_prev, X], dim=0)
                Y_sup = torch.cat([Y_prev, Y], dim=0)

                # Instantiate student and warm-up on real data
                nets = [fn().to(self.device).train() for fn in self.model_fns]
                paramss = [{n: p for n, p in net.named_parameters()} for net in nets]
                # (Re-)initialize momentum buffers each iter if needed
                if self.inner_optimizer == "momentum":
                    velocitiess = [{n: torch.zeros_like(p) for n,p in params.items()} for params in paramss]
                    new_velocs = [{n: v.clone() for n, v in velocitiess[i].items()} for i in range(len(self.model_fns))]
                    
                # Inner-loop on synthetic support
                for t in range(self.T):
                    loss_sups = [0.0] * len(self.model_fns)
                    # Loop over each digit class
                    for c in range(self.num_classes):
                        # real images of class c
                        x_real_c, _ = self.real_loader.dataset.class_sample(c, self.m_per_class)
                        x_real_c = x_real_c.to(self.device)

                        # synthetic images of class c from X_sup/Y_sup
                        mask_c    = (Y_sup == c)
                        x_syn_all = X_sup[mask_c]           # includes both old & new
                        # pick m_per_class of them at random
                        perm = torch.randperm(x_syn_all.size(0), device=self.device)
                        idx  = perm[: self.m_per_class]
                        x_syn_c = x_syn_all[idx]

                        # form the 2m‐sized batch & labels
                        x_cat = torch.cat([x_real_c, x_syn_c], dim=0)         # [2m, C, H, W]
                        y_cat = torch.full((self.m_per_class,), c, device=self.device)
                        y_cat = torch.cat([y_cat, y_cat], dim=0)              # [2m]

                        # augment & forward
                        x_cat = self.aug(x_cat)
                        logitss = [functional_call(net, params, (x_cat,)) for net, params in zip(nets, paramss)]
                        for i, logits in enumerate(logitss):
                            loss_sups[i] += F.cross_entropy(logits, y_cat)

                    # 6) average over classes
                    loss_sups = [loss_sup / float(self.num_classes) for loss_sup in loss_sups]

                    # 7) compute grads & unroll exactly as before
                    gradss = [torch.autograd.grad(loss_sup, params.values(), create_graph=True) for loss_sup, params in zip(loss_sups, paramss)]

                    alpha_t = F.softplus(self.inner_lrs[t])
                    new_params = [{} for _ in range(len(self.model_fns))]
                    for i, (grads, params) in enumerate(zip(gradss, paramss)):
                        for (name,p), g in zip(params.items(), grads):
                            if self.inner_optimizer == "sgd":
                                new_params[i][name] = p - alpha_t * g
                            else:  # momentum
                                v_new = (self.inner_momentum * velocitiess[i][name]
                                        + (1-self.inner_momentum) * g)
                                new_velocs[i][name]   = v_new
                                new_params[i][name]   = p - alpha_t * v_new
                                
                    paramss = new_params
                    if self.inner_optimizer == "momentum":
                        velocitiess = new_velocs      
                        
                    if self.debug: 
                        for i, grads in enumerate(gradss):
                            print(f"     - Model {i+1}: T Loss =        {loss_sups}")
                            print(f"     - Model {i+1}: g_norm =        ", torch.norm(torch.stack([g.norm() for g in grads])).item())
                            print(f"     - Model {i+1}: alpha_t =       ", alpha_t.item())
                                    
                # Meta-evaluate on real batch
                try:
                    x_real, y_real = next(real_iter)
                except StopIteration:
                    real_iter = iter(self.real_loader)
                    x_real, y_real = next(real_iter)
                x_real, y_real = x_real.to(self.device), y_real.to(self.device)
                logitss_real = [functional_call(net, params, (x_real)) for net, params in zip(nets, paramss)]
                
                #    so that each real sample is paired with a synthetic of the same class
                n_real = x_real.size(0)
                # draw a random offset [0,ipc) for *each* real sample
                offsets = torch.randint(0, self.ipc, (n_real,), device=self.device)
                # compute indices into X: if y_real[i] == c, idx = c*ipc + offsets[i]
                syn_indices = y_real * self.ipc + offsets
                x_syn_pair  = X[syn_indices]                 # shape [n_real, C, H, W]
                # L2 penalty on *every* pixel
                recon_loss   = F.mse_loss(x_syn_pair, x_real)
                meta_losses = [F.cross_entropy(logits_real, y_real) + self.regularisation * recon_loss for logits_real in logitss_real]
                
                w = torch.tensor(self.composite_weights, device=meta_losses[0].device, dtype=meta_losses[0].dtype)
                composite_loss = (torch.stack(meta_losses) * w).sum()
                
                stage_losses.append(composite_loss.item())
                
                # Update synthetic X
                syn_opt.zero_grad(); composite_loss.backward() 
                
                if self.debug:
                    print(f"     - K Loss       ={composite_loss}")
                    print("     - ||∇_X meta||  =", X.grad.norm().item())
                    print("     - ΔX norm       =", (syn_opt.param_groups[0]['lr'] * X.grad).norm().item())
                        
                    if k % 20 == 0:
                        self.plot_images(X, self.ipc)
                syn_opt.step()
                
            # Save results
            self.meta_loss_history[f"stage_{stage+1}"] = stage_losses
            self.S_X.append(X.detach().clone())
            self.S_Y.append(Y.detach().clone())

            # Warm-up final student on all synth
            nets = [fn().to(self.device) for fn in self.model_fns]
            for i, (net, theta_0) in enumerate(zip(nets, theta_0s)):
                net.load_state_dict(theta_0)
                opt_inner = SGD(net.parameters(), lr=self.lr_model, momentum=0.9)
                union_X = torch.cat(self.S_X, dim=0)
                union_Y = torch.cat(self.S_Y, dim=0)
                for _ in range(self.warmup_epochs):
                    logits_all = net(union_X)
                    loss_all = F.cross_entropy(logits_all, union_Y)
                    opt_inner.zero_grad(); loss_all.backward(); opt_inner.step()

                theta_0s[i] = net.state_dict()

            if self.debug: 
                for i, net in enumerate(nets):
                    for c in range(10):
                        x_r, y_r = sample_class(dataset, c, 16)
                        x_r, y_r = x_r.to(self.device), y_r.to(self.device)
                        loss_c = F.cross_entropy(net(x_r), y_r)
                        print(f"     - Model {i+1}: Stage {stage}, class {c}, loss {loss_c:.3f}")
                        
        self.final_models = theta_0s
        return self.S_X, self.S_Y
        
    def plot_images(self, X: torch.Tensor, ipc: int, out_path: str = 'assets/debug/synthetic.png'):
        """
        Plot synthetic images grouped by class.

        Args:
            X: Synthetic images tensor of shape (N, C, H, W), ordered by class.
            ipc: Number of images per class to display.
            out_path: Path to save the plotted grid image.
        """
        import os
        from torchvision.utils import make_grid
        from torchvision import transforms
        from PIL import Image
        if self.debug:
            out_path = "assets/debug/synthetic.png"
            os.makedirs("assets/debug", exist_ok=True)
        
        # Ensure X has enough images
        total = ipc * self.num_classes
        imgs = X[:total]

        # Create grid: nrow=ipc images per class -> grid has num_classes rows
        grid = make_grid(imgs, nrow=ipc, normalize=True, scale_each=False)

        # Convert to PIL and optionally resize
        to_pil = transforms.ToPILImage()
        pil_img = to_pil(grid)
        pil_img.save(out_path)
        print(f"Saved synthetic image grid to {out_path}")
        
    def save_model(self, filepaths: list) -> None:
        """
        Save a PyTorch models state_dict to disk.
        
        Args:
            filepaths – list of names where to write the .pth file (e.g. ['checkpoints/model_1.pth', 'checkpoints/model_2.pth'])
        """
        import os
        for final_model, filepath in zip(self.final_models, filepaths):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            torch.save(final_model, filepath)
        print(f"     - models saved to {filepaths}")

    def save_distilled(self, filepath: str) -> None:
        """
        Save the distilled dataset (S_X, S_Y) and meta-loss history.

        Call this *after* distill(), e.g.:

            X_synth, Y_synth = pdd.distill()
            pdd.save_distilled('results/distilled.pth')

        Args:
            filepath – where to write the .pth file
        """
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # make sure distill() has stored these
        data = {
            'X': self.S_X,                    # list of tensors, each [n, C, H, W]
            'Y': self.S_Y,                    # list of tensors, each [n]
            'meta_loss_history': self.meta_loss_history,
            'inner_lrs':       self.inner_lrs.detach().cpu(),
        }
        torch.save(data, filepath)
        print(f"     - distilled dataset & history saved to {filepath}")
    
    def plot_meta_losses(self):
        """
        Plot meta-loss trajectories for each distillation stage.

        Args:
            meta_losses_dict (dict[str, list[float]]):
                keys are stage identifiers (e.g. "1", "2", …),
                values are lists of meta-losses per refinement iteration.
        """
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,5))
        
        # Sort by stage number
        for stage, losses in sorted(self.meta_loss_history.items(), key=lambda x: int(x[0].removeprefix('stage_'))):
            iterations = range(1, len(losses) + 1)
            plt.plot(iterations, losses, marker='o', label=f"Stage {stage}")
        plt.xlabel("Refinement Iteration k")
        plt.ylabel("Meta-Loss")
        plt.title("Meta-Loss Trajectories Across Stages")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("assets/meta-loss-curve.png")