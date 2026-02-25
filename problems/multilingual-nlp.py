SOLUTION = """
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Executing on device: {device}\\n")

# 1. Loading Multilingual Data
print("Loading papluca/language-identification dataset...")
train_dataset = load_dataset("papluca/language-identification", split="train").shuffle(seed=42).select(range(1000))
val_dataset = load_dataset("papluca/language-identification", split="validation").shuffle(seed=42).select(range(400))

unique_labels = sorted(train_dataset.unique("labels"))
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

def adjust_labels(example):
    return {'label': label2id[example['labels']]}

train_dataset = train_dataset.map(adjust_labels)
test_dataset = val_dataset.map(adjust_labels)

train_dataset = train_dataset.remove_columns(["labels"])
test_dataset = test_dataset.remove_columns(["labels"])

# 2. Tokenization with XLM-RoBERTa
model_checkpoint = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)

print("Tokenizing multilingual text...")
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

columns_to_keep = ['input_ids', 'attention_mask', 'label']
tokenized_train = tokenized_train.remove_columns([col for col in tokenized_train.column_names if col not in columns_to_keep])
tokenized_test = tokenized_test.remove_columns([col for col in tokenized_test.column_names if col not in columns_to_keep])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 3. Multilingual Model Initialization
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(unique_labels),
    id2label=id2label,
    label2id=label2id
)
model.to(device)

# 4. Training Setup
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="./xlm-roberta-multilingual-langid",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 5. Training and Multilingual Inference
print("\\nTraining for Language Identification...")
trainer.train()

print("\\nTesting Multilingual Inference:")
samples = [
    "Hello, how are you?",
    "Hola, ¿cómo estás?",
    "Bonjour, comment allez-vous?",
    "Guten Tag, wie geht es Ihnen?",
    "こんにちは、お元気ですか？"
]

model.eval()
with torch.no_grad():
    for text in samples:
        inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
        print(f"Input: {text} --> Predicted Language Code: {model.config.id2label[prediction]}")
""".strip()

DESCRIPTION = "Fine-tune XLM-RoBERTa for multilingual language identification using the HuggingFace Trainer API."
