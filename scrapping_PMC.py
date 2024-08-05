import requests
import pandas as pd
import re
from bs4 import BeautifulSoup

def search_pmc_articles(retmax=500):
    search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pmc&term=open+access&retmax={retmax}&retmode=json"
    response = requests.get(search_url)
    if response.status_code == 200:
        article_ids = response.json().get('esearchresult', {}).get('idlist', [])
        return article_ids
    return []

def fetch_pmc_article_content(article_id):
    fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={article_id}&retmode=xml"
    response = requests.get(fetch_url)
    if response.status_code == 200:
        return response.content
    return None

def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_from_xml(xml_content):
    soup = BeautifulSoup(xml_content, 'lxml')
    paragraphs = soup.find_all('p')
    text = " ".join([para.get_text() for para in paragraphs])
    return preprocess_text(text)

def main():
    # Récupérer les articles
    article_ids = search_pmc_articles()

    # Traiter les articles
    articles_data = []
    for article_id in article_ids:
        content = fetch_pmc_article_content(article_id)
        if content:
            processed_content = extract_text_from_xml(content)
            articles_data.append({'article_id': article_id, 'content': processed_content})
            print(f"Fetched content for article ID: {article_id}.")
        else:
            print(f"Failed to fetch content for article ID: {article_id}.")

    # Convertir en DataFrame
    articles_df = pd.DataFrame(articles_data)
    print(articles_df.head())  # Vérifier les articles

    # Prétraiter l'ensemble des données combinées
    articles_df['content'] = articles_df['content'].apply(preprocess_text)

    # Combiner les ensembles de données
    combined_df = pd.DataFrame({
        'Description': "Article ID: " + articles_df['article_id'],
        'Answer': articles_df['content']
    })

    # Enregistrer ou utiliser le DataFrame combiné pour l'entraînement
    combined_df.to_csv('combined_dataset.csv', index=False)
    print(combined_df.head())  # Vérifier le dataset combiné

if __name__ == "__main__":
    main()
