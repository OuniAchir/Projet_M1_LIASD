import requests
from bs4 import BeautifulSoup
import os
import zipfile
import pandas as pd
import fitz 

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
            # Check if the PDF file exists before adding it to the zip
            if os.path.exists(pdf):
                zipf.write(pdf, os.path.basename(pdf))
    print(f"Created ZIP file: {zip_name}")

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()

def get_arxiv_pdfs_and_create_df(query, max_articles=12000):
    base_url = f"https://arxiv.org/search/?query={query}&searchtype=all&abstracts=show&order=-announced_date_first"
    page = 0
    articles_data = []
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
            abstract = article.find('span', class_='abstract-full').text.strip() if article.find('span', class_='abstract-full') else "No abstract available."
            pdf_link_tag = article.find('a', href=True, text='pdf')

            if pdf_link_tag:
                pdf_url = pdf_link_tag['href']
                if not pdf_url.startswith("https://arxiv.org"):
                    pdf_url = f"https://arxiv.org{pdf_url}"
                filename = f"{title}.pdf".replace(" ", "_").replace("/", "-")
                downloaded_pdf = download_pdf(pdf_url, filename)
                if downloaded_pdf:
                    pdf_content = extract_text_from_pdf(downloaded_pdf)
                    description = f"{title}\n  {abstract}"
                    articles_data.append({'Description': description, 'Answer': pdf_content}) 
                    articles_downloaded += 1

        page += 1  # Move to the next page

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(articles_data)
    return df
    
def create_arrow_dataset(df):
    table = pa.Table.from_pandas(df)
    dataset = Dataset(table)
    return dataset
    
def main():
    # Perform the search and download PDFs
    query = "llm"
    df = get_arxiv_pdfs_and_create_df(query, max_articles=12000)
    return df
        
if __name__ == "__main__":
    dataset = main()
    dataset.to_csv('arxiv_articles.csv', index=False)
    print("Dataset saved to 'arxiv_articles.csv'.")
    dataset = create_arrow_dataset(dataset)
    print(dataset.head())
