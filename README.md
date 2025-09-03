# TensorPlay
A simple deep learning framework designed for educational purposes and small-scale experiments. TensorPlay provides basic building blocks for constructing and training neural networks, including tensor operations, layers, optimizers, and training utilities.

## Features
- **Core Tensor Structure**: Implements a `Tensor` class with automatic gradient computation, supporting basic arithmetic operations and common activation functions (`ReLU`, `Sigmoid`, `Tanh`, `Softmax`, `GELU`).
- **Neural Network Components**: Includes essential layers like `Dense` (fully connected layer) and a `Module` base class for building custom models.
- **Training Utilities**: Provides `DataLoader` for batching data, training / validation loop helpers (`train_on_batch`, `valid_on_batch`), and optimization with `Adam` optimizer.
- **Early Stopping**: Built-in `EarlyStopping` callback to prevent overfitting.
- **Loss Functions**: Implements common loss functions such as `MSE` (Mean Squared Error), `SSE` (Sum of Squared Errors), and NLL (Negative Log Likelihood).

## Basic Usage
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
            correct += acc * batch_size     # 累计正确样本数

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
## Example: KRK Chess Endgame Classification
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
## Limitations and Future Improvements
### Current Limitations
- Only supports 1D `tensors`, no higher-dimensional data (matrices, images).
- Limited layer types (only `Dense` is implemented).
- Basic optimizer support (only `Adam` is available).
- No `GPU` acceleration; all operations are `CPU`-bound.
- Limited debugging tools for computation graphs.

### Planned Improvements
- Add support for n-dimensional tensors (`matrices`, `3D tensors`).
- Implement more layers (`Dropout`, `BatchNorm`).
- Support GPU acceleration via `CUDA` for faster training.
- Improve automatic differentiation efficiency.
- Include more loss functions (`Cross-Entropy`, `MAE`).
- Add visualization tools for computation graphs and training metrics.

## Contributing
Contributions are welcome! Feel free to open issues for bugs or feature requests, or submit pull requests with improvements.

## License
[MIT](https://opensource.org/licenses/MIT)