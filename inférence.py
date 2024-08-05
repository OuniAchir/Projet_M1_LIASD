from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline, logging, AutoModelForSequenceClassification, RagTokenizer, RagTokenForGeneration, RagRetriever
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os,torch
import logging

def setup_model_and_tokenizer(base_model):
  
  # Configuration bitsAndBytes
  bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
  )

  # Chargement du modèle 
  model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config,
  )

  model.gradient_checkpointing_enable()
  model = prepare_model_for_kbit_training(model)

  # Configuration Lora
  config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
  )

  # Appliquer LoRA au modèle
  model = get_peft_model(model, config)
  model.config.use_cache = False
  model.config.pretraining = 1

  # Charger le tokenizer
  tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
  tokenizer.padding_side = 'right'
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.add_eos_token = True
  tokenizer.bos_token, tokenizer.eos_token

  return model, tokenizer

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model after and befor QLora
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainables%: {100 * trainable_params / all_param}"
  )

def generate_response(model, tokenizer, question):
  # Créer un pipeline de génération de texte avec des paramètres ajustés
  pipe_inf = pipeline(
      task="text-generation",
      model=model,
      tokenizer=tokenizer,
      num_return_sequences=1,
      max_length=500,
      do_sample=True,
      temperature=0.7,
  )

  # Générer la réponse
  response = pipe_inf(question)
  
  # Retourner la réponse
  return response[0]['generated_text']

def main():
  
    # Configuration du logging
    logging.basicConfig(level=logging.CRITICAL)

    # Définir le modèle de base
    base_model = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    # Configurer le modèle et le tokenizer
    model, tokenizer = setup_model_and_tokenizer(base_model)

    # Imprimer les paramètres entraînables
    print_trainable_parameters(model)

    # Question complexe pour tester la génération de réponse
    question = "How do the various treatment options for fibromyalgia, to provide a comprehensive management plan for patients, and what role do self-management techniques play in improving patient outcomes?"

    # Générer et afficher la réponse
    response = generate_response(model, tokenizer, question)
    print(response)

if __name__ == "__main__":
    main()
