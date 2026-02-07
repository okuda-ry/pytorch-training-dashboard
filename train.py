from flask import Flask, jsonify, request
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import base64
from io import BytesIO
from PIL import Image
import numpy as np

app = Flask(__name__)

# Global model variable
global_model = None
global_device = None


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
    global global_model, global_device

    print("🔥 GPU training started")

    # Get epochs from request
    data = request.get_json()
    num_epochs = data.get("epochs", 10)
    print(f"Training for {num_epochs} epochs")

    global_device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", global_device)

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

    # Use only 5% of data with balanced class distribution
    train_targets = train_dataset.targets
    val_targets = val_dataset.targets

    # Get balanced indices for training data (5%)
    train_indices = []
    for digit in range(10):
        digit_indices = torch.where(train_targets == digit)[0].tolist()
        # Select 5% of each digit class
        num_samples = max(1, len(digit_indices) // 20)
        train_indices.extend(digit_indices[:num_samples])

    # Get balanced indices for validation data (5%)
    val_indices = []
    for digit in range(10):
        digit_indices = torch.where(val_targets == digit)[0].tolist()
        # Select 5% of each digit class
        num_samples = max(1, len(digit_indices) // 20)
        val_indices.extend(digit_indices[:num_samples])

    print(f"Train samples: {len(train_indices)} (balanced across 10 classes)")
    print(f"Val samples: {len(val_indices)} (balanced across 10 classes)")

    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Model setup
    global_model = SimpleNN().to(global_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(global_model.parameters(), lr=0.001)

    loss_list = []
    val_acc_list = []

    for epoch in range(num_epochs):
        # Training
        global_model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(global_device), target.to(global_device)

            optimizer.zero_grad()
            output = global_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_list.append(round(avg_loss, 3))

        # Validation
        global_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(global_device), target.to(global_device)
                output = global_model(data)
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


@app.route("/predict", methods=["POST"])
def predict():
    global global_model, global_device

    if global_model is None:
        return jsonify({"error": "Model not trained yet. Please train first."}), 400

    data = request.get_json()
    image_data = data.get("image")

    # Decode base64 image
    image_data = image_data.split(",")[1]
    image_bytes = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_bytes)).convert("L")

    # Resize to 28x28
    image = image.resize((28, 28), Image.LANCZOS)

    # Convert to numpy array
    image_array = np.array(image).astype(np.float32)

    # Invert the image (white on black -> black on white)
    # Canvas: white background with black drawing, MNIST: black background with white drawing
    image_array = 255.0 - image_array

    # Normalize to 0-1
    image_array = image_array / 255.0

    # Apply MNIST normalization
    image_array = (image_array - 0.1307) / 0.3081

    # Create tensor
    image_tensor = torch.FloatTensor(image_array).unsqueeze(0).unsqueeze(0)
    image_tensor = image_tensor.to(global_device)

    # Predict with model in eval mode
    global_model.eval()
    with torch.no_grad():
        output = global_model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_digit = torch.max(probabilities, 1)

        # Debug: Print all probabilities
        print(f"Raw output: {output}")
        print(f"Probabilities: {probabilities}")
        print(
            f"Predicted digit: {predicted_digit.item()}, Confidence: {confidence.item()}"
        )

    return jsonify(
        {
            "digit": int(predicted_digit.item()),
            "confidence": float(confidence.item()),
        }
    )


app.run(port=5000)
