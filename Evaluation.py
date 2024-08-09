from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def evaluate_similarity(generated_responses, reference_responses):
    # Initialiser ROUGE
    rouge = Rouge()

    # Vectorize the text data using TF-IDF
    vectorizer = TfidfVectorizer()
    all_responses = generated_responses + reference_responses
    vectorizer.fit(all_responses)
    generated_vectors = vectorizer.transform(generated_responses)
    reference_vectors = vectorizer.transform(reference_responses)

    # Calculer la similarité cosinus pour chaque paire
    cosine_similarities = []
    for i in range(len(generated_responses)):
        similarity = cosine_similarity(generated_vectors[i], reference_vectors[i])[0][0]
        cosine_similarities.append(similarity)
        print(f"Similarité Cosinus pour la paire {i+1} :", similarity) # Affiche chaque similarité

    # Calculer BLEU pour chaque paire
    bleu_scores = []
    for i, (generated_response, reference_response) in enumerate(zip(generated_responses, reference_responses)):
        bleu_score = sentence_bleu([reference_response.split()], generated_response.split())
        bleu_scores.append(bleu_score)
        print(f"BLEU pour la paire {i+1} :", bleu_score) # Affiche chaque score BLEU

    # Calculer ROUGE-N (ROUGE-1, ROUGE-2, ROUGE-L) pour chaque paire
    for i, (generated_response, reference_response) in enumerate(zip(generated_responses, reference_responses)):
        rouge_scores = rouge.get_scores(generated_response, reference_response)
        rouge_1 = rouge_scores[0]['rouge-1']['f']
        rouge_2 = rouge_scores[0]['rouge-2']['f']
        rouge_l = rouge_scores[0]['rouge-l']['f']
        print(f"ROUGE Scores pour la paire {i+1}:")
        print(f"ROUGE-1 (F1) :", rouge_1)
        print(f"ROUGE-2 (F1) :", rouge_2)
        print(f"ROUGE-L (F1) :", rouge_l)

# Exemple d'utilisation
if __name__ == "__main__":
    generated_responses = ["example generated response 1"]
    reference_responses = ["example generated response 1"]

    evaluate_similarity(generated_responses, reference_responses)
