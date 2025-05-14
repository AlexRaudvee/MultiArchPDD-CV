import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST
from models.Model import ConvNet

# ——— Config ———
CKPT_PATH        = "data/Distilled/mm-match_mnist_convnet.pt"
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SYN_BATCH_SIZE   = 32
TEST_BATCH_SIZE  = 256
LR               = 1e-3
EPOCHS_PER_STAGE = 5   # you can tweak this

# ——— 1) Load all synthetic stages ———
data = torch.load(CKPT_PATH, map_location="cpu")
X_list, Y_list = data["X"], data["Y"]
num_stages = len(X_list)

# ——— 2) Instantiate model ———
# infer channels & classes from the *first* stage
C = X_list[0].shape[1]
num_classes = (Y_list[0].argmax(dim=1).max().item() + 1) if Y_list[0].ndim==2 else (Y_list[0].max().item()+1)
model = ConvNet(in_channels=C, num_classes=num_classes).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
model.train()

# ——— 3) Progressive training ———
for stage_idx, (X_syn, Y_syn) in enumerate(zip(X_list, Y_list), start=1):
    # turn soft/one-hot into hard labels if needed
    if Y_syn.ndim == 2:
        y_hard = Y_syn.argmax(dim=1)
    else:
        y_hard = Y_syn.flatten().long()

    ds = TensorDataset(X_syn, y_hard)
    loader = DataLoader(ds, batch_size=SYN_BATCH_SIZE, shuffle=True)

    print(f"\n=== Training on synthetic stage {stage_idx}/{num_stages} (N={len(ds)}) ===")
    for epoch in range(1, EPOCHS_PER_STAGE + 1):
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg = total_loss / len(ds)
        print(f" Stage {stage_idx}, Epoch {epoch}/{EPOCHS_PER_STAGE} → loss {avg:.4f}")

# ——— 4) Evaluate on real MNIST test set ———
transform = transforms.Compose([transforms.ToTensor()])
mnist_test = MNIST("data/mnist", train=False, download=True, transform=transform)
test_loader = DataLoader(mnist_test, batch_size=TEST_BATCH_SIZE, shuffle=False)

model.eval()
correct = 0
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        preds = model(xb).argmax(dim=1)
        correct += (preds == yb).sum().item()

acc = correct / len(mnist_test)
print(f"\nFinal test accuracy on real MNIST: {acc*100:.2f}%")
