import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import re
from Combined_DataFrame import dataset as df
from Scrapping_ArXiv import dataset as df

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to preprocess text
def clean_text(text):
    # Remove HTML tags using BeautifulSoup
    soup = BeautifulSoup(text, "html.parser")
    cleaned_text = soup.get_text(separator=" ")

    # Normalize text
    cleaned_text = cleaned_text.lower()  # Convert to lower case

    # Tokenize and remove stopwords
    tokens = word_tokenize(cleaned_text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    clean_text = ' '.join(tokens)

    return clean_text

def create_dataset(df):
    table = pa.Table.from_pandas(df)
    dataset = Dataset(table)
    return dataset
    
# Apply the preprocessing function to the 'Description' and 'Answer' columns
df['Description'] = df['Description'].apply(clean_text)
df['Answer'] = df['Answer'].apply(clean_text)

# Save the preprocessed DataFrame
df.to_csv('preprocessed_combined_data.csv', index=False)
print("Preprocessed dataset saved to 'preprocessed_combined_data.csv'.")

df['text'] = '[INST]@Enlighten. ' + df['Description'] + '[/INST]' + df['Answer'] + ''
df = df.drop(['Description', 'Answer'], axis=1)
df.head()

# Creates a dataset that is ready to train
dataset = create_arrow_dataset(df)
