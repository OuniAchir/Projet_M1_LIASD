import requests
from bs4 import BeautifulSoup
import os
import zipfile

def download_pdf(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        return filename
    else:
        print(f"Failed to download PDF from {url}")
        return None

def create_zip_from_pdfs(pdf_files, zip_name='articles.zip'):
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        for pdf in pdf_files:
            zipf.write(pdf, os.path.basename(pdf))
    print(f"Created ZIP file: {zip_name}")

def get_arxiv_pdfs(query, max_articles=1200):
    base_url = "https://arxiv.org/search/?query={query}&searchtype=all&abstracts=show&order=-announced_date_first"
    page = 0
    pdf_files = []
    articles_downloaded = 0

    while articles_downloaded < max_articles:
        search_url = f"{base_url}&size=50&start={page * 50}"
        response = requests.get(search_url)

        if response.status_code != 200:
            print("Failed to retrieve search results.")
            break

        soup = BeautifulSoup(response.content, 'html.parser')
        articles = soup.find_all('li', class_='arxiv-result')

        if not articles:
            print("No more articles found.")
            break

        for article in articles:
            if articles_downloaded >= max_articles:
                break

            title = article.find('p', class_='title').text.strip()
            pdf_link_tag = article.find('a', href=True, text='pdf')

            if pdf_link_tag:
                pdf_url = pdf_link_tag['href']
                if not pdf_url.startswith("https://arxiv.org"):
                    pdf_url = f"https://arxiv.org{pdf_url}"
                filename = f"{title}.pdf".replace(" ", "_").replace("/", "-")
                downloaded_pdf = download_pdf(pdf_url, filename)
                if downloaded_pdf:
                    pdf_files.append(downloaded_pdf)
                    articles_downloaded += 1

        page += 1  # Move to the next page

    return pdf_files

# Effectuer la recherche et télécharger les PDF
query = "llama"
pdf_files = get_arxiv_pdfs(query, max_articles=1200)

# Créer un fichier ZIP contenant les PDF téléchargés
if pdf_files:
    create_zip_from_pdfs(pdf_files)
else:
    print("No PDF files were downloaded.")
