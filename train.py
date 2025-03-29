import time
import torch
from transformers import TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from model import get_qwen_model
from dataset_loader import get_dataset, format_example
from huggingface_hub import login  # Import the login function

# Log in to Hugging Face Hub
login(token="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")  # Replace with your actual Hugging Face token

# Load Qwen tokenizer and model
model_id = "Qwen/Qwen2-0.5B-Instruct"
tokenizer, model = get_qwen_model(model_id)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load dataset
dataset = get_dataset()
formatted_data = [format_example(sample) for sample in dataset]

# Training arguments
training_args = TrainingArguments(
    output_dir="./models/trained_qwen",
    per_device_train_batch_size=4,  # Reduce batch size for memory efficiency
    num_train_epochs=3,
    save_strategy="steps",
    save_steps=100,
    logging_steps=50,
    learning_rate=5e-5,
    evaluation_strategy="epoch",
    push_to_hub=True,
)

# Data collator
response_template = "### The response query is:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# Trainer setup
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    formatting_func=lambda x: x["input_text"],  # Ensure input is formatted properly
    data_collator=collator,
    args=training_args,
)

# Push to Hugging Face Hub (after trainer setup)
trainer.push_to_hub("shaleenchordia/qwen-sql-generator")

# Train the model
t1 = time.time()
trainer.train()
print("Training Time:", time.time() - t1)

# Save model locally
trainer.save_model("./models/trained_qwen")

# Push to Hugging Face Hub (again, after training is done)
trainer.push_to_hub("shaleenchordia/qwen-sql-generator")