import numpy as np
from sklearn.metrics import jaccard_score
from sklearn.feature_extraction.text import CountVectorizer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize

# Fonction pour calculer la similarité de Jaccard
def jaccard_similarity(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

# Flatten reference_answers to a list of strings
# Assuming each sublist in reference_answers contains a single string
# reference_answers_flat = [item[0] for item in reference_answers]

# Calculer la similarité de Jaccard
jaccard_similarities = [jaccard_similarity(reference.split(), generated.split()) for reference, generated in zip(answers, generated_responses)]
average_jaccard_similarity = np.mean(jaccard_similarities)
print("Similarité de Jaccard Moyenne :", average_jaccard_similarity)

# Fonction pour calculer la similarité cosinus entre des textes
def cosine_similarity_texts(text1, text2):
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    cosine_sim = np.dot(vectors[0], vectors[1]) / (np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1]))
    return cosine_sim


# Exemple d'utilisation
if __name__ == "__main__":

answers = ["There is currently no cure for fibromyalgia, but treatment options can help manage symptoms. Treatments may include: Medications: These can help reduce pain and improve sleep. Common medications include pain relievers, antidepressants, and anti-seizure drugs. Therapies: Physical therapy can help improve strength and flexibility, while occupational therapy can help make adjustments to work or home environments. Counseling can also be beneficial. Pain Management Clinics: These clinics offer a range of services to help manage pain, including medications, physical therapy, and counseling. Multidisciplinary Pain Management Programs: These programs combine various therapies to help manage pain and improve quality of life. Self-management techniques are crucial for improving outcomes in fibromyalgia. These techniques include: Exercise: Regular physical activity can help reduce pain and improve mood. Activities like walking, swimming, and yoga are beneficial.Healthy Eating: A balanced diet can help manage symptoms and improve overall health. Sleep Hygiene: Good sleep habits can help improve sleep quality and reduce fatigue. Stress Management: Techniques like deep breathing, meditation, and mindfulness can help manage stress, which can exacerbate symptoms. Complementary Treatments: Some patients find relief with acupuncture, massage therapy, or chiropractic care."]
generated_responses = ["The context information provided does mention various treatment options for fibromyalgia and their roles in providing a comprehensive management plan for patients. Physical therapies such as exercise, acupuncture, and heat therapy are mentioned as ways to manage pain and improve mobility. Psychological therapies such as cognitive behavioral therapy (CBT) and mindfulness are also mentioned as helpful in addressing the emotional impact of fibromyalgia. Drug treatments are also discussed, including their benefits and potential side effects.Self-management techniques are also highlighted as important in improving patient outcomes. These include learning about fibromyalgia, finding support groups, encouraging family and friends to learn about the condition, finding ways to talk about feelings, practicing good sleep hygiene, pacing activities, and managing stress.Therefore, the context information does provide an answer to the question about treatment options and self-management techniques for fibromyalgia."]

# Calculer la similarité cosinus
cosine_similarities = [cosine_similarity_texts(reference, generated) for reference, generated in zip(answers, generated_responses)] # Use the flattened list here as well
average_cosine_similarity = np.mean(cosine_similarities)
print("Similarité Cosinus Moyenne :", average_cosine_similarity)

# Calculer BLEU
bleu_scores = [sentence_bleu([reference.split()], generated.split()) for generated, reference in zip(answers, generated_responses)] # Use the flattened list here as well
average_bleu_score = np.mean(bleu_scores)
print("BLEU Moyen :", average_bleu_score)
