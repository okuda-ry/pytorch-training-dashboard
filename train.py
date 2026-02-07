from flask import Flask, jsonify, request
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

app = Flask(__name__)


@app.route("/")
def home():
    return (
        open("index.html", encoding="utf-8").read(),
        200,
        {"Content-Type": "text/html; charset=utf-8"},
    )


@app.route("/style.css")
def css():
    return (
        open("style.css", encoding="utf-8").read(),
        200,
        {"Content-Type": "text/css; charset=utf-8"},
    )


@app.route("/app.js")
def js():
    return (
        open("app.js", encoding="utf-8").read(),
        200,
        {"Content-Type": "application/javascript; charset=utf-8"},
    )


@app.route("/model.png")
def model_image():
    with open("model.png", "rb") as f:
        return f.read(), 200, {"Content-Type": "image/png"}


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


@app.route("/train", methods=["POST"])
def train():
    print("🔥 GPU training started")

    # Get epochs from request
    data = request.get_json()
    num_epochs = data.get("epochs", 10)
    print(f"Training for {num_epochs} epochs")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # Data preparation
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    val_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # Use only 5% of data (half of 10%)
    train_indices = list(range(len(train_dataset) // 20))
    val_indices = list(range(len(val_dataset) // 20))
    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Model setup
    model = SimpleNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    loss_list = []
    val_acc_list = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_list.append(round(avg_loss, 3))

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        val_acc = correct / total
        val_acc_list.append(round(val_acc, 3))

        print(
            f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.3f}, Val Acc: {val_acc:.3f}"
        )

    print("training done")

    return jsonify({"loss": loss_list, "acc": val_acc_list})


app.run(port=5000)
