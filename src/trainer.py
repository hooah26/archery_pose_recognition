from datasets import load_dataset, load_metric
from transformers import ViTForImageClassification
from transformers import TrainingArguments
from transformers import Trainer
from src.utils.utils import *


# 데이터 불러오기
dataset = load_dataset("json", data_files="../data/pose_dataset.json", field="data")
dataset = dataset.class_encode_column("label")
dataset = dataset["train"].train_test_split(test_size=0.2, stratify_by_column="label")
prepared_ds = dataset.with_transform(transform)

# ViT 사전 학습 모델 불러오기
model_name_or_path = 'google/vit-base-patch16-224-in21k'
labels = dataset['train'].features['label'].names
model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)}
).to("cuda")


# 학습 하이퍼파라미터 설정

training_args = TrainingArguments(
  output_dir="./vit-base-beans",
  per_device_train_batch_size=32,
  evaluation_strategy="steps",
  num_train_epochs=50,
  fp16=True,
  save_steps=100,
  eval_steps=100,
  logging_steps=30,
  learning_rate=2e-4,
  save_total_limit=2,
  remove_unused_columns=False,
  push_to_hub=False,
  load_best_model_at_end=True,
  dataloader_num_workers=10
)

# 학습 모듈 불러오기

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds["test"],
    #tokenizer=feature_extractor
)


# 학습 시작
if __name__ == "__main__":
    train_results = trainer.train()

    metrics = trainer.evaluate(prepared_ds['test'])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    trainer.save_model()