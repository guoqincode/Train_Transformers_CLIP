import os
import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel, AdamW
from datasets import Dataset, DatasetDict
from PIL import Image
import random
import json
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm.auto import tqdm
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Fine-tune CLIP with customizable options")
parser.add_argument('--model_name', type=str, required=True, help="Path to pre-trained CLIP model")
parser.add_argument('--save_path', type=str, required=True, help="Path to save the fine-tuned model")
parser.add_argument('--train_json', type=str, required=True, help="Path to training JSON file")
parser.add_argument('--val_json', type=str, required=True, help="Path to validation JSON file")
parser.add_argument('--num_epochs', type=int, default=50, help="Number of epochs")
parser.add_argument('--batch_size', type=int, default=256, help="Batch size")
parser.add_argument('--save_steps', type=int, default=200, help="Number of steps between model saves")
parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help="Number of gradient accumulation steps")
parser.add_argument('--mixed_precision', type=str, choices=['fp16', 'bf16', 'no'], default='fp16', help="Precision of training")
parser.add_argument('--loss_type', type=str, choices=['clip', 'siglip'], default='siglip', help="Choose loss function")
args = parser.parse_args()

# Initialize Accelerator
accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision=args.mixed_precision)

# Configuration
model_name = args.model_name
save_path = args.save_path
num_epochs = args.num_epochs
batch_size = args.batch_size
save_steps = args.save_steps

# Dataset loading function
def load_my_dataset():
    json_path1 = args.train_json
    json_path2 = args.val_json

    def load_data_from_json(json_path):
        image_paths = []
        texts = []
        with open(json_path, 'r') as f:
            data_list = json.load(f)
        for item in data_list:
            image_path = item.get('image', '')
            text = item.get('text', '')
            if os.path.exists(image_path) and text:
                image_paths.append(image_path)
                texts.append(text)
            else:
                print(f"Warning: Missing data. Image path: {image_path}, Text: {text}")
        return {'image_path': image_paths, 'text': texts}

    data1 = load_data_from_json(json_path1)
    data2 = load_data_from_json(json_path2)

    all_image_paths = data1['image_path'] + data2['image_path']
    all_texts = data1['text'] + data2['text']

    total_samples = len(all_image_paths)
    indices = list(range(total_samples))
    random.shuffle(indices)

    val_size = 100
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_image_paths = [all_image_paths[i] for i in train_indices]
    train_texts = [all_texts[i] for i in train_indices]

    val_image_paths = [all_image_paths[i] for i in val_indices]
    val_texts = [all_texts[i] for i in val_indices]

    dataset = DatasetDict({
        'train': Dataset.from_dict({'image_path': train_image_paths, 'text': train_texts}),
        'validation': Dataset.from_dict({'image_path': val_image_paths, 'text': val_texts})
    })
    return dataset

# DataLoader collate_fn
def collate_fn(features):
    images = []
    texts = []

    for f in features:
        image_path = f['image_path']
        text = f['text']
        if os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert('RGB')
                images.append(image)
                texts.append(text)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                continue
        else:
            print(f"Image path does not exist: {image_path}")

    if len(images) == 0:
        return {}

    inputs = processor(
        text=texts,
        images=images,
        padding='max_length',
        truncation=True,
        max_length=77,
        return_tensors='pt'
    )

    return inputs

# SigLIP Loss
def siglip_loss(logits_per_text):
    eye = torch.eye(logits_per_text.size(0), device=logits_per_text.device)
    m1_diag1 = -torch.ones_like(logits_per_text) + 2 * eye
    loglik = F.logsigmoid(m1_diag1 * logits_per_text)
    nll = -torch.sum(loglik, dim=-1)
    loss = nll.mean()
    return loss

# CLIP Loss
def clip_loss(logits_per_text):
    labels = torch.arange(logits_per_text.size(0)).to(logits_per_text.device)
    loss_fct = torch.nn.CrossEntropyLoss()
    return loss_fct(logits_per_text, labels)

# Load dataset
dataset = load_my_dataset()

train_dataloader = DataLoader(
    dataset['train'],
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn
)

eval_dataloader = DataLoader(
    dataset['validation'],
    batch_size=8,
    shuffle=False,
    collate_fn=collate_fn
)

# Load model and processor
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)
model.gradient_checkpointing_enable()

optimizer = AdamW(model.parameters(), lr=5e-5)

# Prepare model and dataloaders
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

total_steps = 0
training_logs = {"step_losses": [], "epoch_losses": []}

# Training loop
for epoch in range(num_epochs):
    model.train()
    losses = []
    for step, batch in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")):
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            pixel_values=batch['pixel_values'],
            return_loss=False
        )
        logits_per_text = outputs.logits_per_text

        # Select loss function based on user input
        if args.loss_type == 'siglip':
            loss = siglip_loss(logits_per_text) / accelerator.gradient_accumulation_steps
        else:
            loss = clip_loss(logits_per_text) / accelerator.gradient_accumulation_steps
        
        accelerator.backward(loss)

        if (step + 1) % accelerator.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

            if accelerator.is_main_process:
                current_loss = loss.detach().float().item() * accelerator.gradient_accumulation_steps
                losses.append(current_loss)
                training_logs["step_losses"].append({"global_step": total_steps+1, "loss": current_loss})
                print(f"Step {total_steps+1}, Loss: {current_loss}")

        total_steps += 1

        if total_steps % save_steps == 0 and accelerator.is_main_process:
            save_folder = os.path.join(save_path, f"step_{total_steps}")
            os.makedirs(save_folder, exist_ok=True)
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(save_folder)
            processor.save_pretrained(save_folder)
            print(f"Model saved at step {total_steps}")

    # Epoch logging
    if accelerator.is_main_process and len(losses) > 0:
        avg_loss = sum(losses) / len(losses)
        training_logs["epoch_losses"].append({"epoch": epoch+1, "avg_loss": avg_loss})
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss}")

# Final model saving
if accelerator.is_main_process:
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    with open(os.path.join(save_path, "training_logs.json"), "w") as f:
        json.dump(training_logs, f, indent=4)
    print("Final model saved. Training logs saved to training_logs.json")
