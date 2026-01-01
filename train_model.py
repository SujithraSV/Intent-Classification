import json
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset # type: ignore
import torch

# 1. Open and load the JSON file
with open('Intent.json', 'r') as file:
    data = json.load(file)

# 2. Get the list of intents
intents_list = data['intents']

# 3. Create a list to hold all our training examples
rows = [] # We will fill this with dictionaries

# 4. Loop through each intent object and extract the data we need
for intent_obj in intents_list:
    intent_name = intent_obj['intent']   # Get the intent label (e.g., 'GreetingUserRequest')
    for example_text in intent_obj['text']: # Get each example phrase
        # Add a new row for each example phrase
        rows.append({'text': example_text, 'intent': intent_name})

# 5. Create the DataFrame
df = pd.DataFrame(rows)

# 6. Let's explore our new, clean DataFrame!
print(f"Total training examples: {len(df)}")
print("\nFirst 10 rows of the DataFrame:")
print(df.head(10))
print("\nNumber of examples per intent:")
print(df['intent'].value_counts())

# Optional: Save this clean DataFrame to a CSV for later use
df.to_csv('clean_intents_dataset.csv', index=False)
print("\nSaved clean data to 'clean_intents_dataset.csv'")

# 7. Prepare the labels - Map intent names to numbers (0, 1, 2, ...)
intent_labels = df['intent'].unique().tolist()
id2label = {id: label for id, label in enumerate(intent_labels)}
label2id = {label: id for id, label in enumerate(intent_labels)}

# Add a new column 'label' with the numerical ID for each intent
df['label'] = df['intent'].map(label2id)

# 8. Split the data into training and validation sets
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Training examples: {len(train_df)}, Validation examples: {len(eval_df)}")

# 9. Load the Tokenizer and Model
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(intent_labels),
    id2label=id2label,
    label2id=label2id
)

# 10. Tokenize the datasets
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# Convert DataFrames to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

# 11. Define Training Arguments (UPDATED WITH CORRECT PARAMETER NAMES)
training_args = TrainingArguments(
    output_dir="./intent_classifier_model",
    learning_rate=2e-5,
    per_device_train_batch_size=8,  # Reduced batch size for smaller dataset
    per_device_eval_batch_size=8,
    num_train_epochs=10,   #num_train_epochs=10,  # Changed from 4 to 10          # Enough to learn from 143 examples
    weight_decay=0.01,
    eval_strategy="epoch",          # CHANGED FROM evaluation_strategy
    save_strategy="epoch",
    logging_dir='./logs',
    load_best_model_at_end=True,
)

# 12. Create Trainer and Train!
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
)

print("\nStarting training... This might take a few minutes.")
trainer.train()
print("Training finished!")

# 13. Save the model for later use
trainer.save_model("./saved_intent_model")
tokenizer.save_pretrained("./saved_intent_model")
print("Model saved to './saved_intent_model'")
"""import json
import pandas as pd
from collections import Counter

# 1. Open and load the JSON file
with open('Intent.json', 'r') as file:
    data = json.load(file)

# 2. Get the list of intents
intents_list = data['intents']

# 3. Create a list to hold all our training examples
rows = [] # We will fill this with dictionaries

# 4. Loop through each intent object and extract the data we need
for intent_obj in intents_list:
    intent_name = intent_obj['intent']   # Get the intent label (e.g., 'GreetingUserRequest')
    for example_text in intent_obj['text']: # Get each example phrase
        # Add a new row for each example phrase
        rows.append({'text': example_text, 'intent': intent_name})

# 5. Create the DataFrame
df = pd.DataFrame(rows)

# 6. Let's explore our new, clean DataFrame!
print(f"Total training examples: {len(df)}")
print("\nFirst 10 rows of the DataFrame:")
print(df.head(10))
print("\nNumber of examples per intent:")
print(df['intent'].value_counts())

# Optional: Save this clean DataFrame to a CSV for later use
df.to_csv('clean_intents_dataset.csv', index=False)
print("\nSaved clean data to 'clean_intents_dataset.csv'")

# ---- ADD THIS CODE TO THE BOTTOM OF YOUR train_model.py FILE ---- #

from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch

# 1. Prepare the labels - Map intent names to numbers (0, 1, 2, ...)
intent_labels = df['intent'].unique().tolist()
id2label = {id: label for id, label in enumerate(intent_labels)}
label2id = {label: id for id, label in enumerate(intent_labels)}

# Add a new column 'label' with the numerical ID for each intent
df['label'] = df['intent'].map(label2id)

# 2. Split the data into training and validation sets
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Training examples: {len(train_df)}, Validation examples: {len(eval_df)}")

# 3. Load the Tokenizer and Model
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(intent_labels),
    id2label=id2label,
    label2id=label2id
)

# 4. Tokenize the datasets
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# Convert DataFrames to Hugging Face Dataset format
from datasets import Dataset
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

# 5. Define Training Arguments
training_args = TrainingArguments(
    output_dir="./intent_classifier_model",
    learning_rate=2e-5,
    per_device_train_batch_size=8,  # Reduced batch size for smaller dataset
    per_device_eval_batch_size=8,
    num_train_epochs=4,             # Enough to learn from 143 examples
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    load_best_model_at_end=True,
)

# 6. Create Trainer and Train!
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
)

print("\nStarting training... This might take a few minutes.")
trainer.train()
print("Training finished!")

# 7. Save the model for later use
trainer.save_model("./saved_intent_model")
tokenizer.save_pretrained("./saved_intent_model")
print("Model saved to './saved_intent_model'")"""