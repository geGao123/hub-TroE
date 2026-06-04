# model.py
import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF  # 修正为标准小写导入


class TokenClassifierOutput:
    """存放模型输出的容器类"""

    def __init__(self, loss=None, logits=None):
        self.loss = loss
        self.logits = logits


class BertNERModel(nn.Module):
    def __init__(self, model_name_or_path: str, num_labels: int, dropout_prob: float = 0.1, use_crf: bool = False):
        super().__init__()
        self.use_crf = use_crf

        # 1. 基础 BERT
        self.bert = BertModel.from_pretrained(model_name_or_path)
        self.dropout = nn.Dropout(dropout_prob)

        # 2. 发射线性头 (Emission Layer)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

        # 3. 条件选择 CRF 层
        if self.use_crf:
            self.crf = CRF(num_tags=num_labels, batch_first=True)
        else:
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: torch.Tensor = None) -> TokenClassifierOutput:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.use_crf:
                # CRF 需要 bool 类型的 mask 矩阵 (1 表示有效 token, 0 表示 padding)
                mask = attention_mask.to(torch.bool)

                # CRF 无法直接识别内部的 -100 标签，计算前将其临时替换为 0 (O 标签)。
                # 配合上面的真实有效 mask 矩阵，Padding 部分的标签不会对梯度产生影响。
                clean_labels = labels.clone()
                clean_labels[clean_labels == -100] = 0

                # torchcrf 返回的是对数似然（负数），求负号转换为正 Loss
                loss = -self.crf(logits, clean_labels, mask=mask, reduction='mean')
            else:
                # 经典的 Softmax + CrossEntropy 路线
                flat_logits = logits.view(-1, logits.shape[-1])
                flat_labels = labels.view(-1)
                loss = self.loss_fn(flat_logits, flat_labels)

        return TokenClassifierOutput(loss=loss, logits=logits)