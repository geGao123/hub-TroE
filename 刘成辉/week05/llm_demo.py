import os
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter

# ==========================================
# 全局模型与训练超参数配置
# ==========================================
D_MODEL = 768
N_HEADS = 12
D_FF = 512
MAX_SEQ_LEN = 512
MODEL_PATH = "llm.pth"
VOCAB_PATH = "vocab.json"


# ==========================================
# 辅助函数：统一处理单行数据（兼容单文本与多轮对话消息格式）
# ==========================================
def format_item(item):
    """
    将 JSONL 中读取的单行对象转化为统一的文本格式。
    如果是消息列表（Messages Format），将其转化为带有角色标识符的多轮对话文本。
    """
    if 'text' in item:
        return item['text']
    elif 'messages' in item:
        formatted_text = ""
        for msg in item['messages']:
            role = msg.get('role', 'user')
            # 格式化角色前缀
            role_prefix = "User" if role == "user" else "Assistant"
            content = msg.get('content', '')
            formatted_text += f"{role_prefix}: {content}\n"
        return formatted_text
    else:
        return str(item)


# ==========================================
# 1. 高效的多头自注意力机制 (带单向/因果掩码兼容)
# ==========================================

class MultiHeadAttention(nn.Module):
    """
    高效的多头自注意力机制 (融合同步投影)
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_k = d_model // n_heads
        self.d_model = d_model
        self.n_heads = n_heads

        # 融合投影：一个线性层直接产生 Q、K、V，速度极快
        self.w_qkv = nn.Linear(d_model, 3 * d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_length, _ = x.shape

        # 融合投影 + chunk 物理切块
        qkv_fused = self.w_qkv(x)
        q, k, v = qkv_fused.chunk(3, dim=-1)

        # 多头拆分 [B, H, L, d_k]
        q = q.view(batch_size, seq_length, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.n_heads, self.d_k).transpose(1, 2)

        # 矩阵乘法计算注意力分数
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)

        if mask is not None:
            # 遮蔽未来位置或 Padding 位置 (mask 值为 0 的地方填充极小值)
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # 头拼接还原
        out_put = (attention @ v).transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        return self.fc_out(out_put)


class TransformerBlock(nn.Module):
    """
    模块化的 Transformer Encoder 层 (采用 Pre-LN 结构与完备的正则化)
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # 前馈网络子层 (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),  # FFN 内部的 Dropout
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # --- 第一子层：MHA + Add & Norm (Pre-Norm 结构) ---
        attn_out = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_out)  # 残差相加并应用 Dropout

        # --- 第二子层：FFN + Add & Norm ---
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)  # FFN 输出端残差前应用 Dropout

        return x


# ==========================================
# 2. Embedding 层
# ==========================================

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, dropout=0.1):
        super().__init__()
        # pad_idx 为 0 的位置不更新梯度
        self.tok_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        seq_len = x.size(1)
        # 生成可学习的位置编码索引
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0).expand_as(x)
        embedding = self.tok_embed(x) + self.pos_embed(pos)
        return self.dropout(self.norm(embedding))


# ==========================================
# 3. 单向因果语言模型 (Causal LM) 封装
# ==========================================

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, max_len, dropout=0.1):
        super().__init__()
        self.embedding = TransformerEmbedding(vocab_size, d_model, max_len, dropout)
        self.transformer = TransformerBlock(d_model, n_heads, d_ff, dropout)
        # 语言模型头
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x, padding_mask=None):
        batch_size, seq_len = x.shape
        out = self.embedding(x)

        # 因果遮蔽矩阵 (Causal Mask)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).view(1, 1, seq_len, seq_len)

        if padding_mask is not None:
            mask = padding_mask * causal_mask
        else:
            mask = causal_mask

        out = self.transformer(out, mask)
        return self.fc(out)


# ==========================================
# 4. 优化适配 Dataset：支持大文件 + 对话多轮格式解析
# ==========================================

class CausalLMDataset(Dataset):
    def __init__(self, data_path, vocab, max_len=128):
        self.data_path = data_path
        self.max_len = max_len
        self.vocab = vocab
        self.offsets = []

        print(f"-> 正在对数据集 {data_path} 构建高速索引映射...")
        with open(data_path, 'rb') as f:
            offset = 0
            for line in f:
                self.offsets.append(offset)
                offset += len(line)

        self.total_lines = len(self.offsets)
        print(f"-> 索引构建完毕。文本总行数: {self.total_lines}")
        self.file_handler = None

    def __len__(self):
        return self.total_lines

    def __getitem__(self, idx):
        if self.file_handler is None:
            self.file_handler = open(self.data_path, 'rb')

        self.file_handler.seek(self.offsets[idx])
        line = self.file_handler.readline().decode('utf-8')

        item = json.loads(line)
        # 调用统一转换函数，将多轮对话转化为 "User: ... \nAssistant: ..." 文本
        text = format_item(item)

        # 字符级分词
        token_ids = [self.vocab.get(char, 1) for char in text]  # 1 为 [UNK]
        token_ids = token_ids[:self.max_len]

        padding_len = self.max_len - len(token_ids)
        token_ids = token_ids + [0] * padding_len

        # Padding Mask 逻辑
        mask = [1 if i < len(text) else 0 for i in range(self.max_len)]

        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.float).unsqueeze(0).unsqueeze(0)  # [1, 1, max_len]
        }


# ==========================================
# 5. 内存友好的流式词表构建函数（同样适配对话格式）
# ==========================================

def get_or_build_vocab(data_path, vocab_cache="vocab.json", max_vocab_size=15000):
    if os.path.exists(vocab_cache):
        print(f"-> 检测到已存在的词表缓存: {vocab_cache}，直接加载...")
        with open(vocab_cache, 'r', encoding='utf-8') as f:
            return json.load(f)

    print(f"-> 未检测到词表缓存。开始流式扫描 {data_path} 构建因果生成词表...")
    char_counter = Counter()

    with open(data_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                # 统一转化，保证 "User:", "Assistant:" 里的英文字符和换行也被正常加入词频统计
                formatted_text = format_item(item)
                char_counter.update(formatted_text)
            except json.JSONDecodeError:
                continue
            if line_num % 50000 == 0:
                print(f"   已流式处理 {line_num} 行数据...")

    vocab = {"[PAD]": 0, "[UNK]": 1}
    for char, _ in char_counter.most_common(max_vocab_size):
        if char not in vocab:
            vocab[char] = len(vocab)

    with open(vocab_cache, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)

    print(f"-> 词表构建完成并缓存。词表大小: {len(vocab)}")
    return vocab


# ==========================================
# 6. 自回归文本生成解码函数
# ==========================================

def generate_text(model, vocab, inv_vocab, prompt, max_gen_len=60, max_seq_len=128, device="cpu"):
    """
    根据给定的前缀提示词自回归预测后续文本
    """
    model.eval()
    input_ids = [vocab.get(char, 1) for char in prompt]
    input_ids = input_ids[-(max_seq_len - 1):]

    with torch.no_grad():
        for _ in range(max_gen_len):
            x = torch.tensor([input_ids], dtype=torch.long, device=device)
            logits = model(x)
            next_token_logits = logits[0, -1, :]

            # 使用贪婪搜索
            next_token_id = torch.argmax(next_token_logits).item()

            if next_token_id == 0:  # 遇到 [PAD] 终止
                break

            input_ids.append(next_token_id)
            if len(input_ids) >= max_seq_len:
                input_ids = input_ids[1:]

    generated = "".join([inv_vocab.get(idx, "[UNK]") for idx in input_ids])
    return generated


# ==========================================
# 7. 智能设备类型检测
# ==========================================
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# ==========================================
# 8. 文本生成训练主循环
# ==========================================

def train_model():
    data_dir = os.getenv("data_dir")
    train_file = f"${data_dir}/Chinese-Qwen3-235B-2507-Distill-data-110k-SFT/qwen3_235b_2507_distill_110k.jsonl"
    if not os.path.exists(train_file):
        print("-> 没有获取数据")
        return

    vocab = get_or_build_vocab(train_file, vocab_cache=VOCAB_PATH)
    inv_vocab = {v: k for k, v in vocab.items()}

    dataset = CausalLMDataset(train_file, vocab=vocab, max_len=MAX_SEQ_LEN)
    device = get_device()
    print(f"💡 当前训练设备: {device}")

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False
    )

    model = TransformerLM(
        vocab_size=len(vocab),
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_ff=D_FF,
        max_len=MAX_SEQ_LEN,
        dropout=0.1
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 15
    print("\n--- 开始单向因果语言模型训练 (多轮对话 & 硬件自适应版) ---")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        total_tokens = 0

        for step, batch in enumerate(dataloader, 1):
            input_ids = batch['input_ids'].to(device)
            mask = batch['mask'].to(device)

            # 自回归训练平移对齐
            x_input = input_ids[:, :-1]
            y_target = input_ids[:, 1:]
            padding_mask = mask[:, :, :, :-1]

            optimizer.zero_grad()
            outputs = model(x_input, padding_mask)

            loss = criterion(outputs.view(-1, len(vocab)), y_target.reshape(-1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x_input.size(0)
            total_tokens += x_input.size(0)

        print(f"Epoch {epoch + 1:02d}/{epochs:02d} | 平均 Loss: {epoch_loss / total_tokens:.4f}")
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"💾 权重已成功保存至 {MODEL_PATH}")

        # 测试自回归文本生成效果
        if (epoch + 1) % 5 == 0 or epoch == 0:
            test_prompt = "User: 写代码的真谛是什么？\nAssistant:"
            generated = generate_text(model, vocab, inv_vocab, test_prompt, max_gen_len=128, max_seq_len=MAX_SEQ_LEN,
                                      device=device)
            print(f"   [测试生成] 输入: \n{test_prompt} \n   -> 生成: \n{generated}\n" + "-" * 40)

    print("\n✅ 训练完成！")


# ==========================================
# 9. 命令行实时交互推理方法 (新加入)
# ==========================================

def interactive_generate():
    """
    命令行交互生成，直接加载训练好的 llm.pth
    """
    if not os.path.exists(VOCAB_PATH):
        print(f"❌ 找不到词表缓存文件 '{VOCAB_PATH}'，请确保你已经完成了模型训练并生成了词表。")
        return

    # 加载词表
    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    inv_vocab = {v: k for k, v in vocab.items()}

    device = get_device()
    print(f"💡 推理加载设备: {device}")

    # 实例化完全相同配置的模型（推理时不使用 Dropout）
    model = TransformerLM(
        vocab_size=len(vocab),
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_ff=D_FF,
        max_len=MAX_SEQ_LEN,
        dropout=0.0
    ).to(device)

    # 检查并加载模型权重
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 找不到模型权重文件 '{MODEL_PATH}'，请先运行训练（--mode train）生成权重。")
        return

    print(f"🔄 正在从 '{MODEL_PATH}' 载入模型权重...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    print("\n" + "=" * 50)
    print("✨ 自制因果 Transformer LLM 交互命令行加载成功！")
    print("👉 输入你的问题，模型将自回归预测回答。")
    print("👉 输入 'exit'、'quit' 或按 Ctrl+C 可安全退出。")
    print("=" * 50)

    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.strip().lower() in ["exit", "quit"]:
                print("👋 退出对话交互。")
                break
            if not user_input.strip():
                continue

            # 拼接输入提示词
            prompt = f"User: {user_input}\nAssistant:"

            # 进行自回归文本生成 (设置长生成窗口 128)
            full_output = generate_text(
                model=model,
                vocab=vocab,
                inv_vocab=inv_vocab,
                prompt=prompt,
                max_gen_len=128,
                max_seq_len=MAX_SEQ_LEN,
                device=device
            )

            # 裁剪获得生成出的回答部分
            response = full_output[len(prompt):]
            print(f"\nAssistant: {response}")
            print("-" * 50)

        except KeyboardInterrupt:
            print("\n👋 退出对话交互。")
            break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="自研单向 Transformer 语言模型训练与生成平台")
    parser.add_argument(
        "--mode",
        type=str,
        default="generate",
        choices=["train", "generate"],
        help="运行模式：'train' (执行训练流程) 或 'generate' (启动交互对话，默认)"
    )
    args = parser.parse_args()

    if args.mode == "train":
        train_model()
    else:
        interactive_generate()