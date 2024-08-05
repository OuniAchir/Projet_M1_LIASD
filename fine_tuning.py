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

def setup_model_and_tokenizer(base_model):
    # Configuration BitsAndBytes
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Charger le modèle
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config,
    )

    # Activer gradient_checkpointing
    model.gradient_checkpointing_enable()

    # Préparer le modèle pour k-bit training
    model = prepare_model_for_kbit_training(model)

    # Configuration LoRA
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Appliquer LoRA au modèle
    model = get_peft_model(model, config)
    model.config.use_cache = False
    model.config.pretraining = 1

    # Charger le tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_eos_token = True
    tokenizer.bos_token = tokenizer.eos_token

    return model, tokenizer

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

    # Charger le dataset
    dataset = load_dataset('path_to_your_dataset')

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
