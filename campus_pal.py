
!pip install sentence-transformers faiss-cpu

from google.colab import files

uploaded = files.upload()

import json

file_path = "camp.json"

with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

print("Sample Data:")
for entry in data[:3]:
    print("Q:", entry["question"])
    print("A:", entry["answer"])
    print("-" * 50)

from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

questions = [entry["question"] for entry in data]
answers = [entry["answer"] for entry in data]

question_embeddings = model.encode(questions, convert_to_numpy=True)

print("Embedding Shape:", question_embeddings.shape)

import faiss

dimension = question_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(question_embeddings)

print("FAISS index created with", index.ntotal, "questions.")

from sentence_transformers import SentenceTransformer, util
import torch

model = SentenceTransformer("all-MiniLM-L6-v2")

questions = [entry["question"] for entry in data]
question_embeddings = model.encode(questions, convert_to_tensor=True)

def get_best_match(user_question):
    user_embedding = model.encode(user_question, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(user_embedding, question_embeddings)
    best_idx = torch.argmax(similarities).item()
    best_question = questions[best_idx]
    return best_question, data[best_idx]["answer"]



!pip install transformers datasets torch
!pip install datasets

from google.colab import files

uploaded = files.upload()

import json
from datasets import Dataset

with open('camp2.json', 'r') as f:
    data = json.load(f)

dataset = Dataset.from_dict({
    'prompt': [item['prompt'] for item in data],
    'completion': [item['completion'] for item in data]
})

dataset[0]

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import json

with open('camp2.json', 'r') as f:
    data = json.load(f)

dataset = Dataset.from_dict({
    'prompt': [item['prompt'] for item in data],
    'completion': [item['completion'] for item in data],
    'label': [0 for _ in data]
})

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

def tokenize_function(examples):
    return tokenizer(examples['prompt'], examples['completion'], truncation=True, padding=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

train_dataset, eval_dataset = tokenized_datasets.train_test_split(test_size=0.2).values()

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import json

with open('camp2.json', 'r') as f:
    data = json.load(f)

dataset = Dataset.from_dict({
    'prompt': [item['prompt'] for item in data],
    'completion': [item['completion'] for item in data],
    'label': [0 if "negative" in item['completion'].lower() else 1 for item in data]
})

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

def tokenize_function(examples):
    return tokenizer(examples['prompt'], truncation=True, padding="max_length", max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

split_datasets = tokenized_datasets.train_test_split(test_size=0.2)
train_dataset = split_datasets["train"]
eval_dataset = split_datasets["test"]

train_dataset = train_dataset.rename_columns({"label": "labels"})
eval_dataset = eval_dataset.rename_columns({"label": "labels"})

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()

from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import json

with open('camp2.json', 'r') as f:
    data = json.load(f)

dataset = Dataset.from_dict({
    'prompt': [item['prompt'] for item in data],
    'completion': [item['completion'] for item in data]
})

t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')

def tokenize_t5_function(examples):
    inputs = [f"generate: {item}" for item in examples['prompt']]
    model_inputs = t5_tokenizer(inputs, padding="max_length", truncation=True, max_length=512)
    labels = t5_tokenizer(text_target=examples['completion'], padding="max_length", truncation=True, max_length=128)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets_t5 = dataset.map(tokenize_t5_function, batched=True)

train_dataset, eval_dataset = tokenized_datasets_t5.train_test_split(test_size=0.2).values()

data_collator = DataCollatorForSeq2Seq(t5_tokenizer, model=t5_model)

training_args_t5 = TrainingArguments(
    output_dir='./results_t5',
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True
)

trainer_t5 = Trainer(
    model=t5_model,
    args=training_args_t5,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator
)

trainer_t5.train()


user_input = input("Ask a question: ")
best_question, best_answer = get_best_match(user_input)
print(f"Answer: {best_answer}")