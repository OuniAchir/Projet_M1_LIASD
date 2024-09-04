import logging
import os
import json
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import torch
from google.colab import drive
from load_model import setup_model_and_tokenizer
from preprocessing import dataset as dataset

def train_model(model, tokenizer, dataset):
    # Hyperparameter
    training_arg = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        save_steps=50,
        logging_steps=1,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
    )

    # Initialisation du SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_arg,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=1024,
        peft_config=config,
        packing=True,
    )

    # Entraîner le modèle
    trainer.train()

    return trainer

def save_model(trainer, save_directory):
    os.makedirs(save_directory, exist_ok=True)
    trainer.save_model(save_directory)
    trainer.model.save_pretrained(save_directory)
    trainer.tokenizer.save_pretrained(save_directory)
    trainer.model.config.save_pretrained(save_directory)

def extract_logs(log_dir):
    logs = []
    for file_name in os.listdir(log_dir):
        if 'trainer_state.json' in file_name:
            with open(os.path.join(log_dir, file_name), 'r') as f:
                trainer_state = json.load(f)
                logs.extend(trainer_state.get('log_history', []))
    return logs

def plot_training_loss(logs, output_file='training_loss.png'):
    steps = [log['step'] for log in logs if 'loss' in log]
    losses = [log['loss'] for log in logs if 'loss' in log]

    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, label='Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.savefig(output_file)
    plt.show()

def main():
    # Configuration du logging
    logging.basicConfig(level=logging.CRITICAL)

    # Définir le modèle de base
    base_model = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    # Configurer le modèle et le tokenizer
    model, tokenizer = setup_model_and_tokenizer(base_model)

    # Entraîner le modèle
    trainer = train_model(model, tokenizer, dataset)

    # Sauvegarder le modèle
    save_directory = '/directory'
    save_model(trainer, save_directory)

    # Extraire et tracer les logs d'entraînement
    logs = extract_logs(save_directory)
    plot_training_loss(logs)

if __name__ == "__main__":
    main()
