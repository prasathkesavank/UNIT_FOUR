pip install transformers datasets scikit-learn torch

from datasets import load_dataset

dataset = load_dataset("imdb")

from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

def preprocess(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

encoded_dataset = dataset.map(preprocess, batched=True)


from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./bert-imdb",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    save_total_limit=1,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"].shuffle(seed=42).select(range(2000)),  # small subset for example
    eval_dataset=encoded_dataset["test"].shuffle(seed=42).select(range(500))
)

trainer.train()
