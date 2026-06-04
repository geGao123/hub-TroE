# trainer.py
import torch
from torch.utils.data import DataLoader
from seqeval.metrics import classification_report, f1_score
from tqdm import tqdm


class NERTrainer:
    def __init__(self, model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader, id2label: dict,
                 device: str = None, grad_accum: int = 1):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.id2label = id2label
        self.grad_accum = grad_accum

    def _train_epoch(self, optimizer, scheduler):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc="Training")

        optimizer.zero_grad()

        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / self.grad_accum

            loss.backward()

            # 满足累积步数或到达 epoch 末尾时更新梯度
            if (step + 1) % self.grad_accum == 0 or (step + 1) == len(self.train_loader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accum
            progress_bar.set_postfix({"loss": f"{(loss.item() * self.grad_accum):.4f}"})

        return total_loss / len(self.train_loader)

    def evaluate(self, dataloader: DataLoader = None, desc: str = "Evaluating"):
        loader = dataloader or self.val_loader
        self.model.eval()
        total_loss = 0
        true_labels, pred_labels = [], []

        with torch.no_grad():
            for batch in tqdm(loader, desc=desc):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_loss += outputs.loss.item()

                # ---- 区分 CRF 和 Softmax 的解码行为 ----
                if getattr(self.model, "use_crf", False):
                    mask = attention_mask.to(torch.bool)
                    # CRF 内部调用维特比算法解码，直接返回预测的标签 ID 二维列表
                    predictions = self.model.crf.decode(outputs.logits, mask=mask)
                else:
                    predictions = torch.argmax(outputs.logits, dim=-1).cpu().tolist()

                # 过滤并回转标签
                for i in range(labels.shape[0]):
                    cpu_labels = labels[i].cpu().tolist()
                    cpu_preds = predictions[i]

                    # 剥离特殊 token 占位的 -100
                    valid_true = [self.id2label[t] for t, p in zip(cpu_labels, cpu_preds) if t != -100]
                    valid_pred = [self.id2label[p] for t, p in zip(cpu_labels, cpu_preds) if t != -100]

                    true_labels.append(valid_true)
                    pred_labels.append(valid_pred)

        eval_loss = total_loss / len(loader)
        f1 = f1_score(true_labels, pred_labels)
        report = classification_report(true_labels, pred_labels, digits=4)

        return eval_loss, f1, report

    def fit(self, epochs: int, optimizer, scheduler, save_path: str = "best_model.pt"):
        best_f1 = 0.0
        for epoch in range(epochs):
            print(f"\n=== Epoch {epoch + 1}/{epochs} ===")
            train_loss = self._train_epoch(optimizer, scheduler)
            val_loss, val_f1, val_report = self.evaluate()

            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
            print("Evaluation Report:\n", val_report)

            if val_f1 > best_f1:
                best_f1 = val_f1
                print(f"--> 检测到性能提升，正在保存最优模型快照: {save_path}")
                torch.save(self.model.state_dict(), save_path)