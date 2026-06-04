# predict.py
import argparse
import torch
from transformers import BertTokenizerFast
from dataset import build_label_schema
from model import BertNERModel


def parse_args():
    parser = argparse.ArgumentParser(description="BERT NER 模型交互式预测")
    parser.add_argument("--use_crf", action="store_true", help="如果训练时用了 CRF，预测时也必须开启")
    parser.add_argument("--bert_path", type=str, default="bert-base-chinese", help="预训练 BERT 模型权重位置")
    parser.add_argument("--model_weight", type=str, required=True,
                        help="训练好的模型权重文件路径 (例如: best_bert_linear_model.pt)")
    parser.add_argument("--max_length", type=int, default=128, help="序列最大长度")
    return parser.parse_args()


def extract_entities(text: str, pred_tags: list) -> dict:
    """
    一个简单的实体解析后处理函数。
    把 ["B-LOC", "I-LOC", "O"] 结合原始文本，切分成结构化的实体字典。
    """
    entities = {}
    current_entity = []
    current_type = None

    for char, tag in zip(text, pred_tags):
        if tag.startswith("B-"):
            # 如果前面已经有一个实体在记录，先保存它
            if current_entity:
                entity_text = "".join(current_entity)
                entities.setdefault(current_type, []).append(entity_text)
                current_entity = []
            current_type = tag.split("-")[1]
            current_entity.append(char)
        elif tag.startswith("I-") and current_type and tag.split("-")[1] == current_type:
            current_entity.append(char)
        else:
            if current_entity:
                entity_text = "".join(current_entity)
                entities.setdefault(current_type, []).append(entity_text)
                current_entity = []
                current_type = None

    # 别忘了收尾最后一个实体
    if current_entity:
        entity_text = "".join(current_entity)
        entities.setdefault(current_type, []).append(entity_text)

    return entities


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 载入标签 Schema 和 Tokenizer
    labels, label2id, id2label = build_label_schema()
    tokenizer = BertTokenizerFast.from_pretrained(args.bert_path)

    # 2. 初始化模型架构并载入训练好的参数权重
    print(f"正在从 {args.model_weight} 载入模型权重...")
    model = BertNERModel(
        model_name_or_path=args.bert_path,
        num_labels=len(label2id),
        use_crf=args.use_crf
    )
    # 加载权重
    model.load_state_dict(torch.load(args.model_weight, map_location=device))
    model.to(device)
    model.eval()
    print("模型加载成功！进入交互模式（输入 'quit' 或 'exit' 退出）。\n" + "=" * 50)

    # 3. 命令行循环交互
    while True:
        try:
            text = input("\n请输入你要测试的句子 >> ").strip()
            if not text:
                continue
            if text.lower() in ["quit", "exit"]:
                print("退出预测。")
                break

            # 文本转换为列表：中文通常按字切分
            tokens = [char for char in text]

            # 4. 数据管道：Tokenizer 编码
            # 注意：预测单条数据不需要 padding，但需要转成 Tensor 加上 Batch 维度
            encoding = tokenizer(
                tokens,
                is_split_into_words=True,
                return_offsets_mapping=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt"
            )

            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            # 5. 模型前向推理与解码
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                if args.use_crf:
                    mask = attention_mask.to(torch.bool)
                    predictions = model.crf.decode(outputs.logits, mask=mask)[0]  # 取出 Batch 的第一条
                else:
                    predictions = torch.argmax(outputs.logits, dim=-1).cpu().tolist()[0]

            # 6. 后处理：剥离 [CLS] 和 [SEP]，还原标签
            word_ids = encoding.word_ids(batch_index=0)
            pred_tags = []

            for idx, word_idx in enumerate(word_ids):
                if word_idx is None:
                    continue  # 忽略 [CLS], [SEP]
                pred_tags.append(id2label[predictions[idx]])

            # 7. 打印好看的结果
            print("-" * 30)
            print("【字标对应】:")
            for c, t in zip(text, pred_tags):
                print(f" {c}({t})", end=" ")
            print("\n")

            print("【识别实体结构化结果】:")
            parsed_entities = extract_entities(text, pred_tags)
            if parsed_entities:
                for ent_type, ent_list in parsed_entities.items():
                    print(f"  📍 {ent_type} 类实体: {ent_list}")
            else:
                print("  ❌ 未检测到任何命名实体。")
            print("-" * 30)

        except Exception as e:
            print(f"发生错误: {e}")


if __name__ == "__main__":
    main()