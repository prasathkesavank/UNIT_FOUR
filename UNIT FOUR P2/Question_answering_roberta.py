!pip install transformers datasets openai --quiet

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments, pipeline
from datasets import load_dataset
import openai
import json

roberta_model_name = "distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(roberta_model_name)
roberta_model = AutoModelForQuestionAnswering.from_pretrained(roberta_model_name)

data = load_dataset("squad")

def prepare(batch):
    tokens = tokenizer(
        batch["question"],
        batch["context"],
        truncation=True,
        max_length=256,
        stride=64,
        padding="max_length",
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )

    starts, ends = [], []
    for i, offset in enumerate(tokens["offset_mapping"]):
        idx = tokens["overflow_to_sample_mapping"][i]
        answer = batch["answers"][idx]
        start_char = answer["answer_start"][0] if answer["answer_start"] else 0
        end_char = start_char + len(answer["text"][0]) if answer["text"] else 0
        seq_ids = tokens.sequence_ids(i)
        c_start = seq_ids.index(1)
        c_end = len(seq_ids) - 1 - seq_ids[::-1].index(1)
        token_start = next((j for j in range(c_start, c_end+1) if offset[j][0] <= start_char < offset[j][1]), c_start)
        token_end = next((j for j in range(c_start, c_end+1) if offset[j][0] < end_char <= offset[j][1]), c_end)
        starts.append(token_start)
        ends.append(token_end)

    tokens["start_positions"] = starts
    tokens["end_positions"] = ends
    tokens.pop("offset_mapping")
    return tokens

train_data = data.map(prepare, batched=True, remove_columns=data["train"].column_names)
train_data.set_format("torch")

args = TrainingArguments(
    output_dir="./roberta_squad",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    learning_rate=3e-5,
    weight_decay=0.01,
    fp16=True,
    logging_steps=50,
    save_strategy="epoch",
    eval_strategy="no"
)

trainer = Trainer(
    model=roberta_model,
    args=args,
    train_dataset=train_data["train"],
)

trainer.train()
trainer.save_model("/content/roberta_squad_model")

roberta_qa = pipeline("question-answering", model="/content/roberta_squad_model", tokenizer=roberta_model_name)

openai.api_key = "YOUR_OPENAI_API_KEY"

def gpt_qa(question, context):
    prompt = f"Answer the question based on the context below:\n\nContext: {context}\nQuestion: {question}\nAnswer:"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

with open("squad_train.jsonl", "w") as f:
    for item in data["train"]:
        for ans in item["answers"]["text"]:
            record = {"prompt": f"Context: {item['context']}\nQuestion: {item['question']}\nAnswer:", 
                      "completion": f" {ans}\n"}
            f.write(json.dumps(record) + "\n")

def ask_question(question, context):
    print("\nRoBERTa Answer:")
    print(roberta_qa({"question": question, "context": context})["answer"])
    print("\nGPT-4 Answer:")
    print(gpt_qa(question, context))

context_text = "SQuAD is a dataset for question answering tasks."
ask_question("What is the purpose of SQuAD?", context_text)
ask_question("Who created SQuAD?", context_text)
