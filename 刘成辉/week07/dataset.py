# dataset.py
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast

# 自动定位数据路径（根据你提供的逻辑）
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data" / "peoples_daily"

class NERDataset(Dataset):
    def __init__(self, recodes: list, tokenizer: BertTokenizerFast, label2id: dict, max_length: int = 128):
        super().__init__()
        self.recodes = recodes
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.recodes)

    def __getitem__(self, index) -> dict:
        item = self.recodes[index]
        tokens = item["tokens"]
        ner_tags = item["ner_tags"]

        # 1. 对文本进行编码 (必须使用 Fast Tokenizer 才能支持 word_ids)
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # 2. 对齐标签 (Label Alignment)
        labels = []
        word_ids = encoding.word_ids(batch_index=0)

        for word_idx in word_ids:
            if word_idx is None:
                # 特殊 Token 标签设为 -100，计算 Loss 时会自动忽略
                labels.append(-100)
            else:
                label_str = ner_tags[word_idx]
                labels.append(self.label2id[label_str])

        # 3. 移除 batch 维度并整理输出
        item_dict = {key: val.squeeze(0) for key, val in encoding.items()}
        item_dict.pop("offset_mapping", None)  # 移除不需要传给模型的 key
        item_dict["labels"] = torch.tensor(labels, dtype=torch.long)

        return item_dict

def build_label_schema() -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """构建 BIO 标签体系，返回 (labels, label2id, id2label)。"""
    d = DATA_DIR
    with open(d / "label_names.json", "r", encoding="utf-8") as f:
        labels = json.load(f)

    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    return labels, label2id, id2label

def load_records(split: str, data_dir: Optional[Path] = None) -> list:
    d = data_dir or DATA_DIR
    with open(d / f"{split}.json", "r", encoding="utf-8") as f:
        return json.load(f)

def build_dataloaders(
    tokenizer: BertTokenizerFast,
    label2id: dict,
    batch_size: int = 32,
    max_length: int = 128,
    data_dir: Optional[Path] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """构建训练/验证/测试 DataLoader，返回 (train_loader, val_loader, test_loader)。"""
    train_records = load_records("train", data_dir)
    val_records = load_records("validation", data_dir)
    test_records = load_records("test", data_dir)

    train_ds = NERDataset(train_records, tokenizer, label2id, max_length)
    val_ds = NERDataset(val_records, tokenizer, label2id, max_length)
    test_ds = NERDataset(test_records, tokenizer, label2id, max_length)

    print(f"数据集规模：训练={len(train_ds)}，验证={len(val_ds)}，测试={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader

