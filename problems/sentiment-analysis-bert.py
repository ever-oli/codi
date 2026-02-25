SOLUTION = """
# BERT FINE-TUNING FOR SENTIMENT ANALYSIS

# !pip install transformers datasets evaluate torch emoji -q

import torch
import emoji
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Executing on device: {device}\\n")


# 1. Data Loading (Tweet Sentiment)

print("Loading tweet_eval sentiment dataset from Hugging Face...")
dataset = load_dataset("tweet_eval", "sentiment")

small_train_dataset = dataset["train"].shuffle(seed=42).select(range(2000))
small_eval_dataset = dataset["validation"].shuffle(seed=42).select(range(500))

print(f"Training subset: {len(small_train_dataset)} rows")
print(f"Validation subset: {len(small_eval_dataset)} rows\\n")


# 2. Text Pre-processing (Handling Emojis)

print("Applying text pre-processing (Demojization)...")

def preprocess_text(example):
    example['text'] = emoji.demojize(example['text'], language='en')
    return example

small_train_dataset = small_train_dataset.map(preprocess_text)
small_eval_dataset = small_eval_dataset.map(preprocess_text)


# 3. Tokenization

print("Loading BERT tokenizer...")
model_checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)

print("Tokenizing datasets...")
tokenized_train = small_train_dataset.map(tokenize_function, batched=True)
tokenized_eval = small_eval_dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# 4. Model Initialization

print("\\nInitializing pre-trained BERT model...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=3
)
model.to(device)


# 5. Training Setup

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="./bert-sentiment-results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir='./logs',
    logging_steps=50,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


# 6. Model Training & Evaluation

print("\\nCommencing Fine-Tuning Process...")
trainer.train()

print("\\nEvaluating the best model on the validation set...")
eval_results = trainer.evaluate()
print(f"Final Validation Accuracy: {eval_results['eval_accuracy'] * 100:.2f}%")


# 7. Inference Example

print("\\nTesting the model with custom text:")
test_sentences = [
    "I absolutely love the new design, it works perfectly! :fire:",
    "This was a terrible waste of my time, the product arrived broken.",
    "It is okay, nothing special but it gets the job done."
]

model.eval()
with torch.no_grad():
    for text in test_sentences:
        processed_text = emoji.demojize(text, language='en')
        inputs = tokenizer(processed_text, return_tensors="pt", truncation=True, max_length=128).to(device)
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probs, dim=-1).item()
        labels_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        print(f"Text: '{text}' --> Prediction: {labels_map[prediction]}")
""".strip()

DESCRIPTION = "Fine-tune BERT for 3-class tweet sentiment analysis using HuggingFace Trainer with emoji demojization preprocessing."
