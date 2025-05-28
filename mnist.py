# mnist_minimal.py
import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms

# ----------------- hyper-parameters -----------------
epochs      = 3
batch_size  = 128
lr          = 1e-3
dev         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ----------------------------------------------------

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),                  # [0,1]
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    ),
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,  # adjust for your VM
    pin_memory=True
)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers  = nn.Sequential(
            nn.Linear(28*28, 256), nn.ReLU(),
            nn.Linear(256, 128),  nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)

model = Net().to(dev)
opt    = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        opt.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        running_loss += loss.item() * x.size(0)
    print(f"Epoch {epoch}: loss={running_loss/len(train_loader.dataset):.4f}")

print("\n✔ Done! Your GPU just trained a tiny neural net.")
