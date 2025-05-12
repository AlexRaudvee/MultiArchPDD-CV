# distillation/pdd.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from collections import OrderedDict
from tqdm.auto import trange


class PDD:
    """
    Progressive Dataset Distillation with naive meta-model matching
    (Algorithm 3 from the report).

    Args:
        model_fn:            callable that returns a fresh nn.Module()
        real_loader:         DataLoader yielding (x_real, y_real) from the true dataset
        image_shape:         tuple (C, H, W)
        num_classes:         number of output classes
        synthetic_size:      number of synthetic examples per P stage
        P:                   number of PDD stages
        K:                   number of synthetic‐refinement iterations per stage
        T:                   number of fine-tuning steps in inner/outer loops
        lr_model:            learning rate for model fine-tuning
        lr_syn_data:         learning rate for synthetic updates
        device:              torch.device (defaults to CUDA if available)
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
        device=None,
    ):
        self.model_fn = model_fn
        self.real_loader = real_loader
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.synthetic_size = synthetic_size
        self.P = P
        self.K = K
        self.T = T
        self.eta = lr_model
        self.eta_syn = lr_syn_data
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def distill(self):
        # 1–2: initialize S ← ∅, θ⁰ ∼ P_θ
        S_X, S_Y = [], []
        real_iter = iter(self.real_loader)
        # random-init theta^0
        theta0 = self.model_fn().to(self.device).state_dict()

        # 3: for i = 1…P do
        for i in range(self.P):

            # 4: (X⁰,Y⁰) ∼ T  → initialize synthetic X,Y
            X = nn.Parameter(
                torch.randn(self.synthetic_size, *self.image_shape, device=self.device)
            )
            Y = nn.Parameter(
                torch.randn(self.synthetic_size, self.num_classes, device=self.device)
            )

            # 5–11: refine synthetic set over K steps
            for k in trange(
                1, self.K + 1,
                desc=f"     [Stage {i+1}/{self.P}] refining synthetic (k)",
                leave=False,
                dynamic_ncols=True
            ):
                # build support S ∪ {(X^{k−1},Y^{k−1})}
                if S_X:
                    X_sup = torch.cat([*S_X, X.detach()], dim=0)
                    Y_sup = torch.cat([*S_Y, F.softmax(Y.detach(), dim=1)], dim=0)
                else:
                    X_sup = X.detach()
                    Y_sup = F.softmax(Y.detach(), dim=1)

                # 6–8: fine-tune model for T steps on support
                model = self.model_fn().to(self.device)
                model.load_state_dict(theta0)
                fast_weights = OrderedDict(model.named_parameters())

                for t in range(1, self.T + 1):
                    logits = functional_call(model, fast_weights, X_sup)
                    loss_sup = -(Y_sup * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
                    grads = torch.autograd.grad(loss_sup, fast_weights.values(), create_graph=True)
                    fast_weights = OrderedDict(
                        (n, p - self.eta * g) for (n, p), g in zip(fast_weights.items(), grads)
                    )

                # 9–10: meta-step → update X,Y by gradient of L_T(θᵀ)
                try:
                    x_real, y_real = next(real_iter)
                except StopIteration:
                    real_iter = iter(self.real_loader)
                    x_real, y_real = next(real_iter)

                x_real = x_real.to(self.device)
                y_real = y_real.to(self.device)

                logits_real = functional_call(model, fast_weights, x_real)
                meta_loss = F.cross_entropy(logits_real, y_real)

                grad_X, grad_Y = torch.autograd.grad(meta_loss, [X, Y])

                with torch.no_grad():
                    X -= self.eta_syn * grad_X
                    Y -= self.eta_syn * grad_Y

            # 12: update S ← S ∪ {(Xᴷ, Yᴷ)}
            S_X.append(X.detach().clone())
            S_Y.append(F.softmax(Y.detach(), dim=1).detach().clone())

            # 13–15: fine-tune model from θ⁰ on S for T steps
            model = self.model_fn().to(self.device)
            model.load_state_dict(theta0)
            for t in range(1, self.T + 1):
                X_all = torch.cat(S_X, dim=0)
                Y_all = torch.cat(S_Y, dim=0)

                logits_all = model(X_all)
                loss_all = -(Y_all * F.log_softmax(logits_all, dim=1)).sum(dim=1).mean()

                model.zero_grad()
                loss_all.backward()
                for p in model.parameters():
                    p.data -= self.eta * p.grad

            # 16: θ⁰ ← θᵀ
            theta0 = model.state_dict()

        # 17: end for
        return S_X, S_Y
