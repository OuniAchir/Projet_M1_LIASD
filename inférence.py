from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline, logging, AutoModelForSequenceClassification, RagTokenizer, RagTokenForGeneration, RagRetriever
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os,torch
import logging
from setup_model import setup_model_and_tokenizer

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
