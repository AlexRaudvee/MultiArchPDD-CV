import torch
import torch.nn as nn
import torch.nn.functional as F
import higher
from torch.optim import SGD, Adam
from tqdm.auto import trange
from torch.func import functional_call

# torch.autograd.set_detect_anomaly(True)

class PDD:
    """
    Progressive Dataset Distillation with naive meta-model matching
    (Algorithm 3 from the report), using `higher` for differentiable inner-loop
    and PyTorch optimizers for both student and synthetic updates.

    Args:
        model_fn:        callable that returns a fresh nn.Module()
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
        model_fn,
        real_loader,
        image_shape,
        num_classes,
        synthetic_size,
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
    ):
        self.model_fn        = model_fn
        self.real_loader     = real_loader
        self.image_shape     = image_shape
        self.num_classes     = num_classes
        self.synthetic_size  = synthetic_size
        self.P               = P
        self.K               = K
        self.T               = T
        self.lr_model        = lr_model
        self.lr_syn_data     = lr_syn_data
        self.syn_optimizer   = syn_optimizer.lower()
        self.syn_momentum    = syn_momentum
        self.inner_optimizer = inner_optimizer.lower()
        self.inner_momentum  = inner_momentum
        self.inner_betas     = inner_betas
        self.inner_eps       = inner_eps
        self.device          = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # storage for monitoring
        self.meta_loss_history = {}  # list of lists per stage
        self.inner_lrs  = nn.Parameter(torch.ones(self.T, device=self.device) * lr_model)

    def distill(self):
        # Initialize distilled support sets
        S_X, S_Y = [], []
        real_iter = iter(self.real_loader)
        # Sample initial model parameters
        theta0 = self.model_fn().to(self.device).state_dict()

        # Main PDD loop over stages
        for stage in range(1, self.P + 1):
            # Initialize synthetic tensors
            X = nn.Parameter(torch.rand(self.synthetic_size, *self.image_shape, device=self.device))
            Y = torch.arange(self.num_classes, device=self.device).repeat_interleave(self.synthetic_size // self.num_classes)
            assert Y.shape[0] == self.synthetic_size
            
            # Synthetic optimizer
            opt_params = [X, self.inner_lrs]
            if self.syn_optimizer == 'sgd':
                syn_opt = SGD(opt_params,
                              lr=self.lr_syn_data,
                              momentum=self.syn_momentum)
            else:
                syn_opt = Adam(opt_params,
                               lr=self.lr_syn_data)

            # K refinement iterations
            self.meta_loss_stage_history = []
            for k in trange(
                1, self.K + 1,
                desc=f"[Stage {stage}/{self.P}] refine syn",
                leave=False,
                dynamic_ncols=True
            ):
                # 1) Build support: S union current synthetic
                if S_X:
                    X_sup = torch.cat([*S_X, X], dim=0)
                    Y_sup = torch.cat([*S_Y, Y], dim=0)
                else:
                    X_sup, Y_sup = X, Y

                # 2) Instantiate student & grab its param dict
                model = self.model_fn().to(self.device)
                model.load_state_dict(theta0)
                params = {name: p for name,p in model.named_parameters()}

                # 3) (Re-)initialize momentum buffers each iter if needed
                if self.inner_optimizer == "momentum":
                    velocities = {n: torch.zeros_like(p) for n,p in params.items()}

                # 4) Unroll T inner‐loop steps with meta‐LR schedule
                for t in range(self.T):
                    logits   = functional_call(model, params, (X_sup,))
                    loss_sup = F.cross_entropy(logits, Y_sup)
                    grads    = torch.autograd.grad(
                        loss_sup, params.values(), create_graph=True
                    )

                    alpha_t = self.inner_lrs[t]
                    new_params = {}
                    for (name,p), g in zip(params.items(), grads):
                        if self.inner_optimizer == "sgd":
                            new_params[name] = p - alpha_t * g
                        else:  # momentum
                            v_new = (self.inner_momentum * velocities[name]
                                     + (1-self.inner_momentum) * g)
                            velocities[name]   = v_new
                            new_params[name]   = p - alpha_t * v_new
                    params = new_params

                # 5) Meta-evaluate on real data (always recompute)
                try:
                    x_real, y_real = next(real_iter)
                except StopIteration:
                    real_iter = iter(self.real_loader)
                    x_real, y_real = next(real_iter)
                x_real, y_real = x_real.to(self.device), y_real.to(self.device)
                logits_real = functional_call(model, params, (x_real,))
                meta_loss   = F.cross_entropy(logits_real, y_real)

                # 6) Record & do exactly one backward per fresh graph
                self.meta_loss_stage_history.append(meta_loss.item())
                syn_opt.zero_grad()
                meta_loss.backward()
                syn_opt.step()

                # 7) Clamp X in no_grad (harmless now)
                with torch.no_grad():
                    X.data.clamp_(0,1)
                    
            # store the metalosses 
            self.meta_loss_history[f"{stage}"] = self.meta_loss_stage_history
            
            # Store final synthetic batch for this stage
            S_X.append(X.detach().clone())
            S_Y.append(Y.detach().clone())


            # Re-train model on full distilled support
            model = self.model_fn().to(self.device)
            model.load_state_dict(theta0)
            for _ in range(self.T):
                X_all = torch.cat(S_X, dim=0)
                Y_all = torch.cat(S_Y, dim=0)
                logits_all = model(X_all)
                loss_all = F.cross_entropy(logits_all, Y_all)
                model.zero_grad()
                loss_all.backward()
                for p in model.parameters():
                    p.data -= self.lr_model * p.grad

            # Update theta0 to the newly trained model
            theta0 = model.state_dict()

        return S_X, S_Y
        
    def plot_meta_losses(self):
        """
        Plot meta-loss trajectories for each distillation stage.

        Args:
            meta_losses_dict (dict[str, list[float]]):
                keys are stage identifiers (e.g. "1", "2", …),
                values are lists of meta-losses per refinement iteration.
        """
        import os 
        os.makedirs("assets", exist_ok=True)
        
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,5))
        # Sort by stage number
        for stage, losses in sorted(self.meta_loss_history.items(), key=lambda x: int(x[0])):
            iterations = range(1, len(losses) + 1)
            plt.plot(iterations, losses, marker='o', label=f"Stage {stage}")
        plt.xlabel("Refinement Iteration k")
        plt.ylabel("Meta-Loss")
        plt.title("Meta-Loss Trajectories Across Stages")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("assets/meta-loss-curve.png")