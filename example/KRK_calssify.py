import random
from TensorPlay import Tensor, Dense, Module, Adam, EarlyStopping
from TensorPlay import DataLoader, train_on_batch, valid_on_batch

# 数据加载
def load_data(path='D:/demo/Shen-main/krkopt.data'):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for i in range(len(lines)):
        row = lines[i].strip().split(',')
        row[-1] = 1 if row[-1] == 'draw' else 0  # 标签：平局为1，否则为0
        for j in range(len(row) - 1):
            if j % 2 == 1:
                row[j] = int(row[j])  # 行坐标（数字）
            else:
                row[j] = ord(row[j]) - ord('a')  # 列坐标（字母转数字）
        lines[i] = row
    return lines

# 数据预处理
def prepare_data(dataset, test_ratio=0.2, val_ratio=0.1):
    """划分训练集、测试集和验证集，格式化为(data, label)元组列表"""
    random.shuffle(dataset)
    split_idx = int(len(dataset) * (1 - test_ratio - val_ratio))
    val_idx = int(len(dataset) * (1 - val_ratio))
    # 转换为(input_features, label)元组列表
    train_data = [(row[:-1], [row[-1]]) for row in dataset[:split_idx]]
    test_data = [(row[:-1], [row[-1]]) for row in dataset[split_idx:val_idx]]
    val_data = [(row[:-1], [row[-1]]) for row in dataset[val_idx:]]
    return train_data, test_data, val_data

# 分类模型
class KRKClassifier(Module):
    def __init__(self, input_size=6):
        super().__init__()
        self.fc1 = Dense(input_size, 32, bias=True)
        self.fc2 = Dense(32, 16, bias=True)
        self.fc3 = Dense(16, 1, bias=True)

    def forward(self, x):
        x = self.fc1(x).relu()
        x = self.fc2(x).relu()
        x = self.fc3(x).sigmoid()
        return x

# 主训练函数
def train(model, loader, val_data, epochs=50, lr=0.01):
    optimizer = Adam(model.params(), lr=lr)
    stoper = EarlyStopping(patience=5, delta=0.1, verbose=True)

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total_samples = 0
        for batch in loader:
            batch_size = len(batch[0])
            total_samples += batch_size

            loss, acc = train_on_batch(model, batch, optimizer)
            total_loss += loss * batch_size  # 累计总损失
            correct += acc * batch_size   # 累计正确率

        avg_loss = total_loss / total_samples
        accuracy = correct / total_samples
        total_loss = 0
        correct = 0
        total_samples = 0
        for batch in val_data:
            batch_size = len(batch[0])
            val_loss, val_acc = valid_on_batch(model, batch)
            total_samples += batch_size
            correct += val_acc * batch_size
            total_loss += val_loss * batch_size

        val_avg_loss = total_loss / total_samples
        val_accuracy = correct / total_samples

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}  Val Loss: {val_avg_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")
        stoper(val_avg_loss, model)
        if stoper.early_stop:
            break


# 测试函数
def test(model, loader):
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        for x, y in zip(batch_x, batch_y):
            pred = 1 if model(x).data[0] > 0.5 else 0
            if pred == y.data[0]:
                correct += 1
            total += 1

    accuracy = correct / total
    print(f"\nTest Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    # 加载并准备数据
    data = load_data()
    train_set, test_set, val_set = prepare_data(data, test_ratio=0.2, val_ratio=0.1)
    print(f"Loaded {len(data)} samples. Train: {len(train_set)}, Test: {len(test_set)}, Val: {len(val_set)}")

    # 创建张量数据加载器
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)  # 测试集不打乱
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

    # 初始化模型并训练
    classifier = KRKClassifier(input_size=6)
    print("Starting training...")
    train(classifier, train_loader, val_loader, epochs=30, lr=0.005)

    # 测试模型
    test(classifier, test_loader)