import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import _T_co


class SampleDataSet(Dataset):

    def __init__(self, number):
        super().__init__()
        self.x = torch.randn(number, 5)
        self.y = torch.argmax(self.x, dim=1)

    def __getitem__(self, index) -> _T_co:
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


class ClassificationModel(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            # nn.Softmax(dim=1),
            nn.Linear(32, output_size),
        )

    def forward(self, x):
        return self.net(x)


def train(device, model_path):
    # 超参数设置
    epochs = 100  # 训练轮数
    batch_size = 50  # 每轮训练样本数
    lr = 0.01  # 学习率

    # 样本个数
    data_set_number = 10000
    # 初始化模型
    model = ClassificationModel(5, 5).to(device)
    # 使用torch的DataLoader，可以简化 batch代码
    data_loader = DataLoader(SampleDataSet(data_set_number), batch_size)

    # 优化器（参数更新器）
    optim = torch.optim.Adam(model.parameters(), lr)
    # 损失函数
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        watch_loss = []
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optim.step()
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
            # print(f"loss:{loss.}")
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, sum(watch_loss) / len(watch_loss)))

    torch.save(model.state_dict(), model_path)


def test_model(device, model_path, test_data_num):
    # 1. 重新实例化模型结构
    model = ClassificationModel(5, 5)

    # 2. 加载保存的权重
    model.load_state_dict(torch.load(model_path, map_location=device))

    # 3. 切换到评估模式
    model.eval()

    # 4. 构造一些新的随机测试数据
    test_data = torch.rand(test_data_num, 5)

    # 5. 测试时不计算梯度，节省内存和算力
    with torch.no_grad():
        outputs = model(test_data)
        # 获取概率最大的索引作为预测结果
        _, predicted = torch.max(outputs, 1)

    # 打印结果对比
    # print("预测结果 (Predicted):", predicted.numpy())
    # 验证逻辑：我们知道数据生成规则是 argmax
    actual = torch.argmax(test_data, dim=1)
    # print("实际标签 (Actual):   ", actual.numpy())

    # 计算准确率
    correct = (predicted == actual).sum().item()
    print(f"测试集准确率: {correct / len(test_data) * 100}%")


if __name__ == '__main__':
    model_path = "model.pth"

    # 在 cuda平台 和 Apple芯片上 启用GPU加速
    # 数据量太小，cpu更快一点
    device = torch.device('cpu')
    # if torch.backends.mps.is_available():
    #     device = torch.device('mps')
    # elif torch.backends.cuda.is_built():
    #     device = torch.device('cuda')
    print(f"使用的device:{device}")

    # 训练
    train(device, model_path)
    # 测试模型
    # test_model(device, model_path, 1000)
