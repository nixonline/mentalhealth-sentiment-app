import pandas as pd
import numpy as np

import os
import glob
import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Helpers
def tokenize(batch):
    return tokenizer(
        batch['statement'],
        padding='max_length',
        truncation=True,
        max_length=256
    )

# Dataset download
path = kagglehub.dataset_download("suchintikasarkar/sentiment-analysis-for-mental-health")
csv_files = glob.glob(os.path.join(path, "*.csv"))

df = pd.read_csv(csv_files[0])
df = df[['statement', 'status']]

# Data cleaning
df = df.dropna(subset=['statement'])

# Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['status'])

num_classes = len(le.classes_)

# Weights
class_weight = {
    0: 1.0,   # Anxiety
    1: 1.0,   # Bipolar
    2: 1.2,   # Depression
    3: 0.6,   # Normal
    4: 1.8,   # Personality disorder
    5: 1.4,   # Stress
    6: 1.5    # Suicidal
}

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['statement'],
    df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

# Tokenization and torch dataset creation
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

train_df = pd.DataFrame({'statement': X_train, 'label': y_train})
test_df  = pd.DataFrame({'statement': X_test,  'label': y_test})

train_ds = Dataset.from_pandas(train_df)
test_ds  = Dataset.from_pandas(test_df)

train_ds = train_ds.map(tokenize, batched=True)
test_ds  = test_ds.map(tokenize, batched=True)

train_ds = train_ds.remove_columns(['statement'])
test_ds  = test_ds.remove_columns(['statement'])

train_ds.set_format("torch")
test_ds.set_format("torch")

# Model initialization
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=num_classes
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_dir="./logs",
    report_to="none"
)

# Multi-class training
model_path = "model_state.pt"

# if os.path.exists(model_path):
#     print("✅ Found model_state.pt — loading weights...")
#     model = AutoModelForSequenceClassification.from_pretrained(
#         "distilbert-base-uncased",
#         num_labels=num_classes
#     )
#     state_dict = torch.load(model_path, map_location=device)
#     model.load_state_dict(state_dict)
# else:
#     print("🚀 model_state.pt not found — training from scratch...")
#     model = AutoModelForSequenceClassification.from_pretrained(
#         "distilbert-base-uncased",
#         num_labels=num_classes
#     )
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_ds,
#         eval_dataset=test_ds,
#         tokenizer=tokenizer
#     )
#     trainer.train()

#     torch.save(model.state_dict(), model_path)

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_ds,
#     eval_dataset=test_ds,
#     tokenizer=tokenizer
# )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer
)
trainer.train()

torch.save(model.state_dict(), model_path)

# Binary model training (Depression vs Suicidal)
depr_code = le.transform(["Depression"])[0]
suic_code = le.transform(["Suicidal"])[0]

binary_df = df[df["label"].isin([depr_code, suic_code])].copy()

binary_df["binary_label"] = binary_df["label"].map({
    depr_code: 0,   # Depression → 0
    suic_code: 1    # Suicidal → 1
})

train_df_bin, eval_df_bin = train_test_split(
    binary_df[["statement", "binary_label"]],
    test_size=0.2,
    random_state=42,
    stratify=binary_df["binary_label"]
)

binary_train_ds = Dataset.from_pandas(train_df_bin)
binary_eval_ds  = Dataset.from_pandas(eval_df_bin)

# Tokenization
binary_train_ds = binary_train_ds.map(tokenize, batched=True)
binary_eval_ds  = binary_eval_ds.map(tokenize, batched=True)

binary_train_ds = binary_train_ds.rename_column("binary_label", "labels")
binary_eval_ds  = binary_eval_ds.rename_column("binary_label", "labels")

binary_train_ds = binary_train_ds.remove_columns(["statement"])
binary_eval_ds  = binary_eval_ds.remove_columns(["statement"])

binary_train_ds.set_format("torch")
binary_eval_ds.set_format("torch")

# Binary model initialization
binary_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

# Binary model arguments
binary_training_args = TrainingArguments(
    output_dir="./binary_results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_dir="./binary_logs",
    report_to="none"
)

# Binary model training
binary_model_path = "binary_model_state.pt"

# if os.path.exists(binary_model_path):
#     print("✅ Found binary_model_state.pt — loading weights...")
#     binary_model = AutoModelForSequenceClassification.from_pretrained(
#         "distilbert-base-uncased",
#         num_labels=2
#     )
#     state_dict = torch.load(binary_model_path, map_location=device)
#     binary_model.load_state_dict(state_dict)
# else:
#     print("🚀 binary_model_state.pt not found — training from scratch...")
#     binary_model = AutoModelForSequenceClassification.from_pretrained(
#         "distilbert-base-uncased",
#         num_labels=2
#     )

#     binary_trainer = Trainer(
#         model=binary_model,
#         args=binary_training_args,
#         train_dataset=binary_train_ds,
#         eval_dataset=binary_eval_ds,
#         tokenizer=tokenizer,
#         callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
#     )
#     binary_trainer.train()

#     torch.save(binary_model.state_dict(), binary_model_path)

# binary_trainer = Trainer(
#     model=binary_model,
#     args=binary_training_args,
#     train_dataset=binary_train_ds,
#     eval_dataset=binary_eval_ds,
#     tokenizer=tokenizer,
#     callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
# )

#     binary_model = AutoModelForSequenceClassification.from_pretrained(
#         "distilbert-base-uncased",
#         num_labels=2
#     )

binary_trainer = Trainer(
    model=binary_model,
    args=binary_training_args,
    train_dataset=binary_train_ds,
    eval_dataset=binary_eval_ds,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)
binary_trainer.train()

torch.save(binary_model.state_dict(), binary_model_path)