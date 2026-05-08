from typing import Any

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import jieba
from collections import Counter
import pickle

# --- 1. 全局配置 ---
MAX_VOCAB_SIZE = 10000 # 设置词表大小 ！如果不限制词表- > 会导致更大模型 + 更慢训练 + 更差泛化 + 更多噪音
MAX_LEN = 256 # 句子最大长度
EMBED_DIM = 256  # 词向量维度
BATCH_SIZE = 64  # 批次大小
EPOCHS = 10  # 训练轮数
LR = 0.001 # 学习率
def get_device():
    if torch.backends.mps.is_available():
        # Apple Silicon GPU 加速后端
        return torch.device("mps")
    elif torch.cuda.is_available():
        # CUDA
        return torch.device("cuda")
    else:
        return torch.device("cpu")

DEVICE = get_device()


# --- 2. 文本处理器 (构建词表) ---
class Vocab:
    def __init__(self, sentences):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        # 1. 统计词频并构建词表
        all_words = []
        for s in sentences:
            all_words.extend(jieba.lcut(str(s)))

        counter = Counter(all_words)
        common_words = counter.most_common(MAX_VOCAB_SIZE - 2)
        for word, _ in common_words:
            self.word2idx[word] = len(self.word2idx)

    def encode(self, text):
        words = jieba.lcut(str(text))
        ids = [self.word2idx.get(w, 1) for w in words]
        # 填充或截断
        if len(ids) < MAX_LEN:
            ids += [0] * (MAX_LEN - len(ids))
        else:
            ids = ids[:MAX_LEN]
        return torch.tensor(ids, dtype=torch.long)


# --- 3. 模型定义  ---
class EmotionLSTMModel(nn.Module):
    def __init__(self, vocab_size, output_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        self.lstm = nn.LSTM(EMBED_DIM, 128, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128 * 2, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        # output 是每一时刻的特征，hidden 是最后的隐藏状态
        _, (hidden, _) = self.lstm(embedded)

        # 拼接双向 LSTM 的最后状态
        cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        return self.fc(cat)

class EmotionRNNModel(nn.Module):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


# --- 4. 数据集类 ---
class CommentDataSet(Dataset):
    def __init__(self, csv_file, vocab):
        df = pd.read_csv(csv_file)
        self.labels = df['label'].values
        self.texts = [vocab.encode(t) for t in df['review']]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.texts[index], torch.tensor(self.labels[index], dtype=torch.long)


# --- 5. 训练主循环 ---
def training():
    # 1.读取原始数据构建词表
    raw_df = pd.read_csv("waimai_10k.csv")
    vocab = Vocab(raw_df['review'])
    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)  # 存下整个工具类实例

    print(f"词表已保存，大小为: {len(vocab.word2idx)}")

    # 2.实例化数据集和加载器
    dataset = CommentDataSet("waimai_10k.csv", vocab)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3.实例化模型
    model = EmotionLSTMModel(len(vocab.word2idx), 2).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print(f"开始训练，设备: {DEVICE}")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)

            # 标准四步：清零、前传、反传、更新
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        val_loader = DataLoader(dataset, batch_size=100, shuffle=True)
        acc = evaluate(model, val_loader)  # 测试本轮模型结果
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.4f}, acc: {acc}")

    torch.save(model.state_dict(), "model.pt")
    print("训练完成，模型已保存。")

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, sample):
    model.eval()

    batch_x, batch_y = next(iter(sample))

    x, y = batch_x.to(DEVICE), batch_y.to(DEVICE)
    print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), len(y) - sum(y)))
    correct = 0

    with torch.no_grad():
        outputs = model(x)  # [batch, 2]

        # 3. 正确取预测类别
        preds = torch.argmax(outputs, dim=1)

        correct = (preds == y).sum().item()

    acc = correct / len(y)

    print("正确预测个数：%d, 正确率：%f" % (correct, acc))
    return acc

def test_model(texts, model_path, vocab):
    # 1. 设备选择
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")

    # 2. 初始化模型并加载权重
    # 注意：vocab_size 必须与训练时保存的词表大小一致
    vocab_size = len(vocab.word2idx)
    model = EmotionLSTMModel(vocab_size, 2).to(device)

    # 加载保存的权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 切换到评估模式（关闭 Dropout 等）

    print(f"正在使用设备: {device} 进行推断...\n")

    results = []
    with torch.no_grad():  # 推断时不计算梯度，节省内存和算力
        for text in texts:
            # 3. 数据预处理
            input_tensor = vocab.encode(text).unsqueeze(0).to(device)  # [1, MAX_LEN]

            # 4. 前向传播
            outputs = model(input_tensor)

            # 5. 获取概率和预测结果
            probs = F.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probs, dim=1)

            label = "好评 (1)" if prediction.item() == 1 else "差评 (0)"
            results.append((text, label, confidence.item()))

    return results

def test():
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    test_samples = [
        "送餐速度非常快，菜品还是热乎的，味道很正宗！",
        "太难吃了，肉全是肥的，以后再也不点了。",
        "分量十足，包装也很精美",
        "等了一个半小时才到，面都糊了",
        "这也能开餐馆？",
        "以后还点这家",
        "很一般",
        "中规中矩",
        "狗都不吃",
        "这是最后一次点这家了"
    ]

    # 调用测试函数
    predictions = test_model(test_samples, "model.pt", vocab)

    # 打印结果
    print(f"{'原话':<30} | {'预测结果':<10} | {'置信度'}")
    print("-" * 60)
    for text, label, conf in predictions:
        print(f"{text[:25]:<30} | {label:<10} | {conf:.2%}")

if __name__ == '__main__':
    training()
    test()


