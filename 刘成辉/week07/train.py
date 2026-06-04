# train.py
import argparse
from pathlib import Path
import torch
from torch.optim import AdamW
from transformers import BertTokenizerFast, get_linear_schedule_with_warmup

# 完美引入我们分层写好的三个本地模块
from dataset import build_label_schema, build_dataloaders
from model import BertNERModel
from trainer import NERTrainer

# 设定你项目默认的 BERT 模型权重名称或本地路径
DEFAULT_BERT_PATH = "bert-base-chinese"


def parse_args():
    parser = argparse.ArgumentParser(description="训练基于 BERT-NER 序列标注模型")
    parser.add_argument("--use_crf", action="store_true", help="是否激活顶部 CRF 条件随机场层")
    parser.add_argument("--bert_path", type=str, default=DEFAULT_BERT_PATH, help="预训练 BERT 模型权重位置")
    parser.add_argument("--epochs", type=int, default=3, help="总训练周期数")
    parser.add_argument("--batch_size", type=int, default=32, help="单次批处理样本量")
    parser.add_argument("--max_length", type=int, default=128, help="序列最大裁剪/填充长度")
    parser.add_argument("--lr", type=float, default=2e-5, help="BERT 基础层编码器的基准学习率")
    parser.add_argument("--head_lr_mult", type=float, default=5.0,
                        help="下游随机初始化层(Classifier/CRF)的学习率放大倍数")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="学习率热身步数比例")
    parser.add_argument("--grad_accum", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout 丢弃率")
    return parser.parse_args()


def main():
    # 1. 解析注入的命令行参数
    args = parse_args()
    print("=" * 50)
    print(" 🚀 正在初始化 NER 训练引擎，当前注入配置如下:")
    for k, v in vars(args).items():
        print(f"   -> {k}: {v}")
    print("=" * 50)

    # 2. 读取标签体系定义
    labels, label2id, id2label = build_label_schema()

    # 3. 实例化 Fast Tokenizer 准备进行对齐编码
    tokenizer = BertTokenizerFast.from_pretrained(args.bert_path)

    # 4. 构建数据装载器
    train_loader, val_loader, test_loader = build_dataloaders(
        tokenizer=tokenizer,
        label2id=label2id,
        batch_size=args.batch_size,
        max_length=args.max_length
    )

    # 5. 基于 nn.Module 的自定义神经网络组装
    model = BertNERModel(
        model_name_or_path=args.bert_path,
        num_labels=len(label2id),
        dropout_prob=args.dropout,
        use_crf=args.use_crf
    )

    # 6. 配置分层差分学习率策略 (避免预训练 BERT 权重被摧毁，同时确保顶层快速收敛)
    head_params = []
    base_params = []
    for name, param in model.named_parameters():
        if "classifier" in name or "crf" in name:
            head_params.append(param)
        else:
            base_params.append(param)

    grouped_optimizer_parameters = [
        {"params": base_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr * args.head_lr_mult}
    ]
    optimizer = AdamW(grouped_optimizer_parameters, weight_decay=0.01)

    # 7. 配置 Warmup 调度器
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # 8. 实例化系统控制类，开始全自动生命周期管理
    save_filename = "best_bert_crf_model.pt" if args.use_crf else "best_bert_linear_model.pt"

    trainer = NERTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        id2label=id2label,
        grad_accum=args.grad_accum
    )

    # 执行训练与开发集实时验证
    trainer.fit(epochs=args.epochs, optimizer=optimizer, scheduler=scheduler, save_path=save_filename)

    # 9. 训练闭环收尾：拉取最优存档进行终极盲测
    print("\n" + "*" * 30 + " 训练完毕，正在加载最优权重执行盲测集检验 " + "*" * 30)
    model.load_state_dict(torch.load(save_filename, map_location=trainer.device))
    test_loss, test_f1, test_report = trainer.evaluate(dataloader=test_loader, desc="Final Testing")
    print(f"🎯 最终测试集表现 -> 损失值: {test_loss:.4f} | 实体块级别 F1: {test_f1:.4f}")
    print("完整测试分类报告:\n", test_report)


if __name__ == "__main__":
    main()