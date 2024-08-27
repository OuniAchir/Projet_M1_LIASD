import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import re

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

# Load the combined DataFrame
df = pd.read_csv('combined_data.csv')

# Apply the preprocessing function to the 'Description' and 'Answer' columns
df['Description'] = df['Description'].apply(clean_text)
df['Answer'] = df['Answer'].apply(clean_text)

# Display the first few rows of the DataFrame to verify the result
print(df.head())

# Save the preprocessed DataFrame
df.to_csv('preprocessed_combined_data.csv', index=False)
print("Preprocessed dataset saved to 'preprocessed_combined_data.csv'.")
