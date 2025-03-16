from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, GPT2Config
import json
from torch.utils.data import Dataset
import torch

# Custom Dataset Class
class DialogueDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt = self.data[idx]['prompt']
        response = self.data[idx]['response']
        
        # Tokenizing the prompt with padding
        inputs = self.tokenizer(
            prompt, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        
        # Tokenizing the response with padding
        labels = self.tokenizer(
            response, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        
        input_ids = inputs['input_ids'].squeeze()  # Remove the batch dimension
        labels_ids = labels['input_ids'].squeeze()  # Remove the batch dimension

        # Replacing padding tokens in labels with -100 (so they are ignored during loss computation)
        labels_ids[labels_ids == self.tokenizer.pad_token_id] = -100

        return {"input_ids": input_ids, "labels": labels_ids}

# Load and prepare dataset
with open('dialogue_dataset.json') as f:
    data = json.load(f)

# Initialize GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have a pad token, using eos_token as padding

# Create dataset instance
dataset = DialogueDataset(data, tokenizer)

# Initialize GPT-2 model from scratch using GPT2Config
config = GPT2Config(vocab_size=tokenizer.vocab_size)  # Define configuration with vocabulary size
model = GPT2LMHeadModel(config)  # Initialize the model with the configuration

# Training arguments
training_args = TrainingArguments(
    output_dir='./backend/models/gpt2/',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    logging_dir='./logs',
    logging_steps=100,
    save_steps=500,
    save_total_limit=2,
    evaluation_strategy="no",  # Setting evaluation_strategy to "no" to avoid eval dataset requirement
    warmup_steps=200,
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=True,  # Enable mixed precision training for speed
)

# Trainer initialization
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('./backend/models/gpt2/')
tokenizer.save_pretrained('./backend/models/gpt2/')

print("Model fine-tuned and saved.")
