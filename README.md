# 🧠 Deep Learning Research Portfolio

A real-time neural network training visualization web application using PyTorch and Flask. Train a model on the MNIST dataset and watch the training progress unfold with interactive graphs.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)
![Flask](https://img.shields.io/badge/Flask-2.0+-green?logo=flask)
![License](https://img.shields.io/badge/License-MIT-yellow)

## ✨ Features

- 🚀 **Real-time Graph Visualization** - Watch training loss and validation accuracy update dynamically
- ⚙️ **Adjustable Epochs** - Change the number of epochs directly from the web UI (1-50)
- 🎯 **MNIST Dataset** - 28×28 pixel handwritten digit images
- 💨 **Fast Training** - Uses 5% of the dataset for quick demos
- 📊 **Detailed Metrics** - Track training loss and validation accuracy throughout training
- 🎨 **Beautiful UI** - Modern gradient design with smooth animations
- 💻 **GPU Support** - Automatically uses CUDA if available

## 🛠️ Installation

### Requirements

- Python 3.8+
- PyTorch
- Flask
- torchvision

### Setup

1. Clone or download this repository:

```bash
cd your-project-directory
```

2. Install dependencies:

```bash
pip install torch torchvision flask
```

3. Run the application:

```bash
python train.py
```

4. Open your browser and navigate to:

```
http://127.0.0.1:5000
```

## 🎮 How to Use

1. **Set Epochs**: Enter the desired number of epochs (1-50) in the "Epochs" field
2. **Start Training**: Click the "Train Model" button
3. **Watch Progress**: Real-time graphs update as the model trains
4. **Analyze Results**: View training loss (left) and validation accuracy (right)

## 🏗️ Model Architecture

### Neural Network Configuration

```
Input Layer:  784 neurons (28×28 flattened image)
    ↓
Hidden Layer 1: 128 neurons + ReLU + Dropout(0.2)
    ↓
Hidden Layer 2: 64 neurons + ReLU + Dropout(0.2)
    ↓
Output Layer: 10 neurons (digits 0-9)
```

### Training Configuration

| Parameter     | Value            |
| ------------- | ---------------- |
| Optimizer     | Adam             |
| Learning Rate | 0.001            |
| Loss Function | CrossEntropyLoss |
| Batch Size    | 64               |
| Data Split    | 5% of MNIST      |

## 📊 Expected Results

As training progresses, you'll observe:

- **Training Loss**: Steadily decreases (red line)
- **Validation Accuracy**: Gradually increases (cyan line)
- **Epochs**: Customizable from 1 to 50

### Example Output (10 epochs)

```
Epoch 1/10 - Loss: 0.598, Val Acc: 0.945
Epoch 2/10 - Loss: 0.275, Val Acc: 0.965
Epoch 3/10 - Loss: 0.205, Val Acc: 0.972
...
```

## 📁 Project Structure

```
project/
├── train.py          # Flask server & training logic
├── index.html        # Web UI
├── app.js            # Client-side visualization
├── style.css         # Styling
├── README.md         # This file
└── data/             # MNIST dataset (auto-downloaded)
```

## 🚀 Performance Notes

- **First Run**: Dataset downloads automatically (~11MB)
- **Training Speed**: ~20-30 seconds for 10 epochs on GPU
- **Data Usage**: Uses 5% of MNIST (3,000 training, 1,000 validation images)

## 🎓 Educational Value

This project demonstrates:

- ✅ Neural network design and implementation
- ✅ Training loops and optimization
- ✅ Real-time data visualization
- ✅ Web-based ML interfaces
- ✅ PyTorch and Flask integration

## 🔧 Customization

### Modify Number of Epochs

The UI allows changing epochs dynamically without code changes.

### Change Model Architecture

Edit the `SimpleNN` class in `train.py`:

```python
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # Modify layers here
```

### Adjust Data Percentage

In `train.py`, change the divisor in these lines:

```python
train_indices = list(range(len(train_dataset) // 20))  # Change 20 to adjust
val_indices = list(range(len(val_dataset) // 20))
```

## 📱 Browser Compatibility

- Chrome/Chromium ✅
- Firefox ✅
- Safari ✅
- Edge ✅

## ⚠️ Important Notes

- This demo uses only 5% of MNIST for speed. For production, use the full dataset.
- GPU is optional but recommended for faster training.
- Accuracy plateaus around 97-98% due to the simple architecture and limited data.

## 📚 Technologies Used

| Technology | Purpose                       |
| ---------- | ----------------------------- |
| PyTorch    | Deep learning framework       |
| Flask      | Web server                    |
| Chart.js   | Real-time graph visualization |
| HTML/CSS   | Web interface                 |
| JavaScript | Client-side logic             |

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Report bugs
- Suggest improvements
- Submit pull requests

## 📄 License

This project is licensed under the MIT License.

## 👨‍💻 Author

Created as a Deep Learning research portfolio project.

---

**Happy Learning! 🎯**

For questions or issues, please open an issue on GitHub.
