<div align="center">
    <p>
        <img src="../docs/images/banner%20(4).png" alt="TensorPlay Banner">
    </p>

[![stars](https://img.shields.io/github/stars/TensorPlay?color=ccf)](https://github.com/bluemoon-o2/TensorPlay)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/tensorplay?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLUE&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/tensorplay)
![PyPI - Downloads](https://img.shields.io/pypi/dm/TensorPlay?labelColor=blue)
![GitHub last commit](https://img.shields.io/github/last-commit/bluemoon-o2/TensorPlay?labelColor=teal)

![python](https://img.shields.io/badge/python-3.8~3.11-aff.svg)
![os](https://img.shields.io/badge/os-win%2C%20linux%2C%20mac-pink.svg)
![hardware](https://img.shields.io/badge/hardware-cpu-red.svg)
[![License](https://img.shields.io/badge/license-MIT-green)](../LICENSE)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/bluemoon-o2/TensorPlay)

**TensorPlay is a learner-friendly, DIY-ready deep learning framework designed for educational purposes and small-scale experiments.**
</div>

# TensorPlay
[![Framework](https://img.shields.io/badge/TensorPlay-0.1.2-orange)](https://pypi.org/project/TensorPlay/)
![Accuracy](https://img.shields.io/badge/User%20Friendliness-🌞-green)
![Pure Python](https://img.shields.io/badge/Pure_Python-✓-purple)
![Out of box](https://img.shields.io/badge/Out_of_the_box-🔧-maroon)

> [!TIP]
> The TensorPlay 0.1.2 Technical Report is now available. See details at: [TensorPlay 0.1.2 Documentation](https://www.welog.me/article/TensorPLay)

**TensorPlay** provides basic building blocks for constructing and training neural networks, including tensor operations, layers, optimizers, and training utilities.

**To start quickly**, simply run the following command in your terminal:
```bash
pip install tensorplay
```

## TensorPlay 0.1.2 Core Features
- **Core Tensor Structure**: Implements a `Tensor` class with automatic gradient computation, supporting basic arithmetic operations and common activation functions (`ReLU`, `Sigmoid`, `Tanh`, `Softmax`, `GELU`).
- **Neural Network Components**: Includes essential layers like `Dense` (fully connected layer) and a `Module` base class for building custom models.
- **Training Utilities**: Provides `DataLoader` for batching data, training / validation loop helpers (`train_on_batch`, `valid_on_batch`), and optimization with `Adam` optimizer.
- **Early Stopping**: Built-in `EarlyStopping` callback to prevent overfitting.
- **Loss Functions**: Implements common loss functions such as `MSE` (Mean Squared Error), `SSE` (Sum of Squared Errors), and NLL (Negative Log Likelihood).

<div align="center">
    <p>
        <img src="../docs/images/banner%20(1).png" alt="TensorPlay Architecture">
    </p>
    <p>
        <img src="../docs/images/structure.png" alt="TensorPlay Architecture">
    </p>
</div>

### 📣 Recent updates
<details>
<summary><strong>🔥🔥2025.09.08: TensorPlay 0.1.2 Released</strong></summary>

- **Bug Fixes:**
  - ✍️Solve the `Initializer Error` in the `Dense` class.
  - 🧮 Added the missing `graph` method in the `Module` class.

- **Documentation Improvements:**
  - 🌐Added a demo `KRK_classify` to the Examples.
  - 🎯Added the raw original edition `TensorPlay v0.1` for comparison and research purposes.

</details>

| Dependency Packages | Corresponding Functionality                                                                                                                                       |
|---------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `Numpy`             | Provides fundamental numerical computing capabilities, supporting tensor operations, array manipulations, and underpinning the core mathematical computations of the framework. |
| `scikit-learn`      | Offers utilities for data preprocessing, evaluation metrics, and integration with classical machine learning workflows, enhancing TensorPlay's compatibility with standard ML pipelines. |

## ⚡Basic Usage
### 1. Define a Model
Create custom models by inheriting from `Module` and defining the forward pass:
```python
from TensorPlay import Module, Dense

class MyModel(Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = Dense(input_size, hidden_size)
        self.fc2 = Dense(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x).relu()  # Apply ReLU activation after first layer
        x = self.fc2(x).sigmoid()  # Apply Sigmoid for binary classification
        return x
```
### 2. Prepare Data
Use `DataLoader` to handle batching and shuffling:
```python
from TensorPlay import DataLoader

# Assume data is a list of (input_features, label) tuples
train_data = [(x1, y1), (x2, y2), ...]
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
```
### 3. Train the Model
Train the model using the `train_on_batch` function. A typical training loop includes batch training, validation, and early stopping judgment:
```python
from TensorPlay import Adam, EarlyStopping

def train(model, loader, val_data, epochs=50, lr=0.01):
    optimizer = Adam(model.params(), lr=lr)
    stoper = EarlyStopping(patience=5, delta=0.1, verbose=True)

    for epoch in range(epochs):
        # Training phase
        total_loss = 0
        correct = 0
        total_samples = 0
        for batch in loader:
            batch_size = len(batch[0])
            total_samples += batch_size

            # Batch training and obtaining loss and accuracy
            loss, acc = train_on_batch(model, batch, optimizer)
            total_loss += loss * batch_size  # 累计总损失
            correct += acc * batch_size  # 累计正确样本数

        # Calculate the average metrics of the training set
        avg_loss = total_loss / total_samples
        accuracy = correct / total_samples

        # Verification phase
        total_loss = 0
        correct = 0
        total_samples = 0
        for batch in val_data:
            batch_size = len(batch[0])
            val_loss, val_acc = valid_on_batch(model, batch)
            total_samples += batch_size
            correct += val_acc * batch_size
            total_loss += val_loss * batch_size

        # Calculate the average metrics of the validation set
        val_avg_loss = total_loss / total_samples
        val_accuracy = correct / total_samples

        # Print training information
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}  "
              f"Val Loss: {val_avg_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")
        
        # Early stopping check
        stoper(val_avg_loss, model)
        if stoper.early_stop:
            break
```
### 4. Evaluate the Model
```python
def test(model, test_loader):
    correct = 0
    total = 0
    for batch_x, batch_y in test_loader:
        for x, y in zip(batch_x, batch_y):
            # Make predictions on a single sample
            # Set the threshold according to the task type 
            # (taking binary classification as an example here)
            pred = 1 if model(x).data[0] > 0.5 else 0  # For binary classification
            if pred == y.data[0]:
                correct += 1
            total += 1
    print(f"Test Accuracy: {correct/total:.4f}")
```
## 🧩Example: KRK Chess Endgame Classification
The `demo/KRK_classify.py` script demonstrates classifying chess endgame positions (King-Rook-King) as either a draw or not. Key steps:
1. **Data Loading**: Parses `krkopt.data` into numerical features.
2. **Data Preparation**: Splits data into training, validation, and test sets.
3. **Model Definition**: Uses a 3-layer fully connected network (`KRKClassifier`).
4. **Training**: Uses `Adam` optimizer with early stopping.
5. **Evaluation**: Computes test accuracy.

Run the example:
```bash
python demo/KRK_classify.py
```

## 🔄Limitations and Future Improvement
### Current Limitations
- No `GPU` acceleration; all operations are `CPU`-bound.
- Limited debugging tools for computation graphs.

### Planned Improvements
- Support GPU acceleration via `CUDA` for faster training.
- Improve automatic differentiation efficiency.

## 👩‍👩‍👧‍👦Contributing
Contributions are welcome! Feel free to open issues for bugs or feature requests, or submit pull requests with improvements.

## 📄 License
This project is released under the [MIT license](../LICENSE).  

## 🔗Links
* [Source code and issue tracker](https://github.com/bluemoon-o2/TensorPlay)
* [PyPI release](https://pypi.org/project/TensorPlay/)
* [Documentation](https://www.welog.me/article/TensorPLay)
