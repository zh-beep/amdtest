# mnist_minimal_guarded.py
import os, multiprocessing as mp
import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
import time



# ------------------------------------------------------------------
# 0️⃣  Safe start-method for DataLoader workers
#      macOS & Windows need "spawn"; Linux can keep the faster "fork".
#      We do it *before* importing torch.multiprocessing internals.
# ------------------------------------------------------------------
if os.name != "posix" or "darwin" in os.sys.platform:
    mp.set_start_method("spawn", force=True)   # macOS / Windows
# (Linux defaults to "fork", which is fine for CUDA / ROCm)

# ------------------------------------------------------------------
# 1️⃣  Device + backend detection
# ------------------------------------------------------------------
if torch.cuda.is_available():          # NVIDIA
    dev = torch.device("cuda")
elif torch.version.hip and torch.cuda.is_available():   # AMD ROCm
    dev = torch.device("cuda")         # ROCm shows up as cuda:0
elif torch.backends.mps.is_available():                 # Apple-Silicon
    dev = torch.device("mps")
else:
    dev = torch.device("cpu")

print(f"Using device: {dev}")

# pin_memory only helps when the destination is a *CUDA/HIP* device
pin_mem = dev.type == "cuda"

# ------------------------------------------------------------------
# 2️⃣  Hyper-parameters (feel free to tweak)
# ------------------------------------------------------------------
epochs      = 3
batch_size  = 128
lr          = 1e-3
num_workers = 4 if batch_size >= 64 else 2   # crude heuristic

# ------------------------------------------------------------------
# 3️⃣  Build DataLoader (inside a function to avoid re-import chaos)
# ------------------------------------------------------------------
def get_loader():
    return torch.utils.data.DataLoader(
        datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=num_workers > 0,   # avoid extra forks each epoch
    )

# ------------------------------------------------------------------
# 4️⃣  Simple MLP model
# ------------------------------------------------------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers  = nn.Sequential(
            nn.Linear(28*28, 256), nn.ReLU(),
            nn.Linear(256, 128),   nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.layers(self.flatten(x))

# ------------------------------------------------------------------
# 5️⃣  Main training loop
# ------------------------------------------------------------------
def main():
    start_time = time.perf_counter()
    train_loader = get_loader()

    model   = Net().to(dev)
    opt     = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred  = model(x)
            loss  = loss_fn(pred, y)
            loss.backward()
            opt.step()
            running_loss += loss.item() * x.size(0)

        print(f"Epoch {epoch}: loss={running_loss/len(train_loader.dataset):.4f}")

    end_time = time.perf_counter()
    full = end_time - start_time
    print(f"\n✔ Done on {dev}! Your tiny net finished training in {full}.")

# ------------------------------------------------------------------
# 6️⃣  Python entry-point guard
# ------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        # Common first-run hiccup on macOS: fallback to single-worker
        if "start a new process before the current process has finished" in str(e):
            print("⚠️  Multiprocessing issue detected. Retrying with num_workers=0…")
            num_workers = 0
            main()
        else:
            raise
