import torch                                    # torch imports 
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import SGD, Adam
from tqdm.auto import trange
from torch.func import functional_call

from utils.func import sample_class             # custom imports 
from distillation.augment import DiffAug

from typing import List, Tuple, Dict            # typing imports
# torch.autograd.set_detect_anomaly(True)

class PDDMultiBranch:
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
        syn_optimizer="adam",
        syn_momentum=0.9,
        inner_optimizer="sgd",
        inner_momentum=0.9,
        inner_betas=(0.9, 0.999),
        inner_eps=1e-8,
        device=None,
        z_init_std=0.05
    ):
        self.model_fns          = model_fns
        self.real_loader        = real_loader
        self.C, self.H, self.W  = image_shape
        self.num_classes        = num_classes
        self.ipc                = ipc
        self.P                  = P
        self.K                  = K
        self.T                  = T
        self.lr_model           = lr_model
        self.lr_syn_data        = lr_syn_data
        self.syn_optimizer      = syn_optimizer.lower()
        self.syn_momentum       = syn_momentum
        self.inner_optimizer    = inner_optimizer.lower()
        self.inner_momentum     = inner_momentum
        self.inner_betas        = inner_betas
        self.inner_eps          = inner_eps
        self.device             = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.debug              = True
        self.warmup_steps       = 5
        self.smooth_kernel      = 1
        self.z_init_std         = z_init_std
        self.m_per_class        = 2

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
    
    def _consistency_loss(self,
        Xs: List[torch.Tensor],
        Ys: List[torch.Tensor],
        lam: float,
    ) -> torch.Tensor:
        """
        Given lists of M synthetic‐data tensors Xs[i] ∈ R^{B×C×H×W} and
        synthetic‐label tensors Ys[i] ∈ R^{B}, compute:

        L_cons = λ * (
            ∑_{i=1..M} || Xs[i] – mean(Xs) ||²
        + ∑_{i=1..M} || Ys[i] – mean(Ys) ||²
        )

        If do_alignment=True, then after computing L_cons, we in‐place align
        each Xs[i], Ys[i] by pulling them λ-proportion of the way toward the mean,
        and finally overwrite all branches with that same aligned mean.

        Returns:
        L_cons  (scalar, with gradient w.r.t. every Xs[i], Ys[i])
        """

        M = len(Xs)
        assert M == len(Ys), "must have same number of Xs and Ys"

        # 1) compute branch‐mean
        mean_X = sum(Xs) / M      # elementwise mean of each branch
        mean_Y = sum(Ys) / M

        # 2) consistency loss
        loss_X = sum(torch.sum((X - mean_X).pow(2)) for X in Xs)
        loss_Y = sum(torch.sum((Y - mean_Y).pow(2)) for Y in Ys)
        L_cons = lam * (loss_X + loss_Y)

        return L_cons
    
    def _alignment(self,
                   Xs: List[torch.Tensor],
                   Ys: List[torch.Tensor], 
                   lam: float):
        
        M = len(Xs)
        assert M == len(Ys), "must have same number of Xs and Ys"
        
        mean_X = sum(Xs) / M      # elementwise mean of each branch
        mean_Y = sum(Ys) / M
        
        # align and synchronize in‐place, under no_grad
        X_aligned = (1/M)*(sum(Xs)) - lam*(sum(X - mean_X for X in Xs))
        # Y_aligned = (1/M)*(sum(Ys)) - lam*(sum(Y - mean_Y for Y in Ys))
        Y_aligned = Ys[0] # coz all ys are the same.
        
        return X_aligned, Y_aligned

        
    def distill(self):
        # Initial student checkpoint
        dataset = self.real_loader.dataset
        theta_0s = [fn().to(self.device).state_dict() for fn in self.model_fns]
        real_iter = iter(self.real_loader)

        for stage in range(self.P):
            # Initialize new synthetic batch
            Xs = [nn.Parameter(torch.randn(self.ipc * self.num_classes, self.C, self.H, self.W, device=self.device) * 0.05) for _ in range(len(self.model_fns))]
            Ys = [torch.arange(self.num_classes, device=self.device).repeat_interleave(self.ipc) for _ in range(len(self.model_fns))]

            # Meta optimizer (only X)
            syn_opt = SGD(Xs, lr=self.lr_syn_data, momentum=0.9) if False  \
                else Adam(Xs, lr=self.lr_syn_data)
            stage_losses = [[] for _ in range(len(self.model_fns))]

            # Prepare previous synthetic
            
            if self.S_X:
                X_prev = torch.cat(self.S_X, dim=0).detach()
                Y_prev = torch.cat(self.S_Y, dim=0).detach()
            else:
                X_prev = torch.empty(0, self.C, self.H, self.W, device=self.device)
                Y_prev = torch.empty(0, dtype=torch.long, device=self.device)

            for k in trange(self.K, desc=f"Stage {stage+1}/{self.P}", leave=False):
                # Build support set
                X_sups = [torch.cat([X_prev, X], dim=0) for X in Xs]
                Y_sups = [torch.cat([Y_prev, Y], dim=0) for Y in Ys]

                # Instantiate student and warm-up on real data
                nets = [fn().to(self.device).train() for fn in self.model_fns]
                paramss = [{n: p for n, p in net.named_parameters()} for net in nets]
                
                # (Re-)initialize momentum buffers each iter if needed
                if self.inner_optimizer == "momentum":
                    velocitiess = [{n: torch.zeros_like(p) for n,p in params.items()} for params in paramss]
                
                loss_sups   = [.0 for _ in range(len(self.model_fns))]
                new_paramss = [{} for _ in range(len(self.model_fns))]
                for i, (X_sup, Y_sup) in enumerate(zip(X_sups, Y_sups)):
                    loss_sup = 0.0
                    # Inner-loop on synthetic support
                    for t in range(self.T):
                        # Loop over each digit class
                        for c in range(self.num_classes):
                            # --- real images of class c ---
                            x_real_c, _ = self.real_loader.dataset.class_sample(c, self.m_per_class)
                            x_real_c = x_real_c.to(self.device)

                            # --- synthetic images of class c from X_sup/Y_sup ---
                            mask_c    = (Y_sup == c)
                            x_syn_all = X_sup[mask_c]           # includes both old & new
                            # pick m_per_class of them at random
                            perm = torch.randperm(x_syn_all.size(0), device=self.device)
                            idx  = perm[: self.m_per_class]
                            x_syn_c = x_syn_all[idx]

                            # --- form the 2m‐sized batch & labels ---
                            x_cat = torch.cat([x_real_c, x_syn_c], dim=0)         # [2m, C, H, W]
                            y_cat = torch.full((self.m_per_class,), c, device=self.device)
                            y_cat = torch.cat([y_cat, y_cat], dim=0)              # [2m]

                            # --- augment & forward ---
                            x_cat = self.aug(x_cat)
                            logits = functional_call(nets[i], paramss[i], (x_cat,))
                            loss_sup += F.cross_entropy(logits, y_cat)

                        # 6) average over classes
                        loss_sup = loss_sup / float(self.num_classes)
                        loss_sups[i] = loss_sup
                        
                        # 7) compute grads & unroll exactly as before
                        grads= torch.autograd.grad(loss_sup, paramss[i].values(), create_graph=True)

                        alpha_t = F.softplus(self.inner_lrs[t])

                        for (name,p), g in zip(paramss[i].items(), grads):
                            if self.inner_optimizer == "sgd":
                                new_paramss[i][name] = p - alpha_t * g
                            else:  # momentum
                                v_new = (self.inner_momentum * velocitiess[i][name]
                                        + (1-self.inner_momentum) * g)
                                velocitiess[i][name]   = v_new
                                new_paramss[i][name]   = p - alpha_t * v_new
                                    
                        paramss[i] = new_paramss[i]

                        if self.debug: 
                            print(f"     - Model {i+1}: T Loss ={loss_sup}")
                            print(f"     - Model {i+1}: g_norm =", torch.norm(torch.stack([g.norm() for g in grads])))
                            print(f"     - Model {i+1}: alpha_t =", alpha_t.item())
                                        
                
                # Meta-evaluate on real batch
                try:
                    x_real, y_real = next(real_iter)
                except StopIteration:
                    real_iter = iter(self.real_loader)
                    x_real, y_real = next(real_iter)
                x_real, y_real = x_real.to(self.device), y_real.to(self.device)
                logitss_real = [functional_call(net, params, (x_real)) for net, params in zip(nets, paramss)]
                _meta_losses = [F.cross_entropy(logits_real, y_real) * 50000 for logits_real in logitss_real]
                
                # compute consistency loss
                cons_loss = self._consistency_loss(Xs, Ys, lam=0.1)
                
                meta_losses = [meta_loss + cons_loss for meta_loss in _meta_losses]
                stage_losses.append([meta_loss.item() for meta_loss in meta_losses])

                # prepare the loss for backward pass
                meta_loss = torch.stack(meta_losses).mean()
                
                # Update synthetic Xs
                syn_opt.zero_grad(); meta_loss.backward(retain_graph=(i < len(meta_losses) - 1)); syn_opt.step()
                    
                # align branches
                X_aligned, Y_aligned = self._alignment(Xs, Ys, lam=0.1)

                
                if self.debug:
                    print(f"     - K Loss      ={meta_loss}")
                    for i, X in enumerate(Xs):
                        print(f"     - Model {i+1}: ||∇_X meta|| =", X.grad.norm().item())
                        print(f"     - Model {i+1}: ΔX norm  =", (syn_opt.param_groups[0]['lr'] * X.grad).norm().item())
                        
                    if k % 20 == 0:
                        self.plot_images(X_aligned, self.ipc)
                
            # Save results
            self.meta_loss_history[f"stage_{stage+1}"] = stage_losses
            self.S_X.append(X_aligned.detach().clone())
            self.S_Y.append(Y_aligned.detach().clone())

            # Warm-up final student on all synth
            union_X = torch.cat(self.S_X, dim=0)
            union_Y = torch.cat(self.S_Y, dim=0)
            nets = [fn().to(self.device) for fn in self.model_fns]
            for i, (net, theta_0) in enumerate(zip(nets, theta_0s)):
                net.load_state_dict(theta_0)
                opt_inner = SGD(net.parameters(), lr=self.lr_model, momentum=0.9) if self.inner_optimizer == "momentum" \
                    else Adam(net.params(), lr=self.lr_model)
                for _ in range(self.T):
                    logits_all = net(union_X)
                    loss_all = F.cross_entropy(logits_all, union_Y)
                    opt_inner.zero_grad(); loss_all.backward(); opt_inner.step()

                theta_0s[i] = net.state_dict()

            if self.debug: 
                for i, net in enumerate(nets):
                    for c in range(10):
                        x_r, y_r = sample_class(dataset, c, 16)
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
        print(f"     - Saved synthetic image grid to {out_path}")
        
    def save_model(self, filepaths: List[str]) -> None:
        """
        Save a PyTorch models state_dict to disk.
        
        Args:
            filepaths – list of names where to write the .pth file (e.g. ['checkpoints/model_1.pth', 'checkpoints/model_2.pth'])
        """
        import os
        for final_model, filepath in zip(self.final_models, filepaths):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            torch.save(final_model, filepath)
        print("[Distillator]:") 
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
        print("[Distillator]:") 
        print(f"     - distilled dataset & history saved to {filepath}")
    
    def plot_meta_losses(self):
        """
        Plot meta-loss trajectories for each distillation stage and each model,
        using different colors for stages and different line styles for models.
        
        Expects:
            self.meta_loss_history : dict[str, list[list[float]]]
                Keys are stage identifiers (e.g. "stage_1"), values are lists
                of per-model loss lists, e.g. [[losses_model1], [losses_model2], ...].
        """
        import matplotlib.pyplot as plt
        
        # Pre-define up to 3 line styles for different models
        line_styles = ['-', '--', ':']
        
        # Prepare figure
        plt.figure(figsize=(10, 6))
        
        # Sort stages by numeric suffix
        sorted_items = sorted(
            self.meta_loss_history.items(),
            key=lambda x: int(x[0].split('_')[-1])
        )
        
        # Generate a distinct color for each stage
        colors = plt.cm.tab10.colors  # up to 10 distinct colors
        
        for stage_idx, (stage, per_model_losses) in enumerate(sorted_items):
            color = colors[stage_idx % len(colors)]
            
            for model_idx, losses in enumerate(per_model_losses):
                iterations = range(1, len(losses) + 1)
                style = line_styles[model_idx % len(line_styles)]
                
                plt.plot(
                    iterations,
                    losses,
                    linestyle=style,
                    color=color,
                    marker='o',
                    label=f"{stage} – Model {model_idx+1}"
                )
        
        plt.xlabel("Refinement Iteration $k$")
        plt.ylabel("Meta-Loss")
        plt.title("Meta-Loss Trajectories Across Stages and Models")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("assets/meta-loss-curve.png")
        plt.close()