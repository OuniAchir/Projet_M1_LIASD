import pandas as pd
import re
import urllib.request
import xml.etree.ElementTree as ET
from datasets import load_dataset
import pyarrow as pa
from datasets import Dataset

# Cleaning Functions
def clean_text(text):
    """Remove extra spaces and unwanted characters from text."""
    return re.sub(r'\s+', ' ', text.strip())

def clean_description(description):
    """Clean the 'Description' field specifically."""
    return clean_text(description).lstrip('Q. ')

def clean_answer(answer):
    """Clean the 'Answer' field specifically."""
    return clean_text(answer).replace('->', '')

# Data Loading and Processing Functions
def load_and_process_data1():
    data1 = load_dataset("ruslanmv/ai-medical-chatbot")
    df1 = pd.DataFrame(data1["train"][::])
    df1 = df1[["Description", "Doctor"]].rename(columns={"Description": "Description", "Doctor": "Answer"})
    df1['Description'] = df1['Description'].apply(clean_description)
    df1['Answer'] = df1['Answer'].apply(clean_answer)
    return df1

def load_and_process_data2():
    data2 = pd.read_csv('https://raw.githubusercontent.com/mistralai/cookbook/9f912bd07794afc067f523f6ecb1f2c3fbbd5092/data/Symptom2Disease.csv')
    df2 = pd.DataFrame(data2)
    df2 = df2[["label", "text"]].rename(columns={"label": "Answer", "text": "Description"})
    df2['Description'] = df2['Description'].apply(clean_description)
    df2['Answer'] = df2['Answer'].apply(clean_answer)
    return df2

def load_and_process_data3():
    url = 'https://trec.nist.gov/data/qa/2017_LiveQA/med-qs-and-reference-answers.xml'
    response = urllib.request.urlopen(url)
    data3 = response.read()
    root = ET.fromstring(data3)
    data_3 = []

    for question in root.findall('.//NLM-QUESTION'):
        message = clean_text(question.find('MESSAGE').text) if question.find('MESSAGE') is not None else ''

        answer_text = ''
        for answer in question.findall('Answer'):
            answer_text += clean_text(answer.text) if answer.text else ''

        for answer in question.findall('AnswerURL'):
            answer_text += f" URL: {clean_text(answer.text)}" if answer.text else ''

        for answer in question.findall('Comment'):
            answer_text += f" Comment: {clean_text(answer.text)}" if answer.text else ''

        data_3.append({'Description': message, 'Answer': answer_text})

    df3 = pd.DataFrame(data_3)
    return df3

def load_and_process_data4():
    data4 = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split='train')
    df4 = pd.DataFrame(data4)
    df4 = df4[["input", "output"]].rename(columns={"input": "Description", "output": "Answer"})
    df4['Description'] = df4['Description'].apply(clean_description)
    df4['Answer'] = df4['Answer'].apply(clean_answer)
    return df4

def extract_data(entry):
    """Helper function to extract and format data from dataset 5."""
    qtext = entry['qtext']
    answers = " ".join([f"{ans['aid']}. {ans['atext']}" for ans in entry['answers']])
    Description = f"{qtext} {answers}"
    Answer = entry['ra']
    return {'Description': Description, 'Answer': Answer}

def load_and_process_data5():
    data5 = load_dataset("head_qa", "en", split='train')
    data5 = [extract_data(row) for row in data5]
    df5 = pd.DataFrame(data5)
    df5['Description'] = df5['Description'].apply(clean_text)
    df5['Answer'] = df5['Answer'].apply(lambda x: clean_text(str(x)))
    return df5

def convert_to_str(entry):
    """Convert list entries to strings."""
    if isinstance(entry, list):
        return " ".join(entry)
    return str(entry)

def load_and_process_data6():
    data6 = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split='train')
    df6 = pd.DataFrame(data6)
    df6['context'] = df6['context'].apply(lambda x: convert_to_str(x['contexts']))
    df6['Description'] = df6['context'] + " " + df6['question']
    df6['Description'] = df6['Description'].apply(clean_text)
    df6['Answer'] = df6['final_decision'] + ", " + df6['long_answer'].apply(clean_text)
    df6 = df6[['Description', 'Answer']]
    return df6

def NaN_none(entry):
    return str(entry) if pd.notna(entry) else ""

def load_and_process_data7():
    data7 = load_dataset("openlifescienceai/medmcqa", split='train')
    df7 = pd.DataFrame(data7)
    df7['question'] = df7['question'].apply(NaN_none).astype(str)
    df7['opa'] = df7['opa'].apply(NaN_none).astype(str)
    df7['opb'] = df7['opb'].apply(NaN_none).astype(str)
    df7['opc'] = df7['opc'].apply(NaN_none).astype(str)
    df7['opd'] = df7['opd'].apply(NaN_none).astype(str)
    df7['subject_name'] = df7['subject_name'].apply(NaN_none).astype(str)
    df7['topic_name'] = df7['topic_name'].apply(NaN_none).astype(str)
    df7['cop'] = df7['cop'].apply(NaN_none).astype(str)
    df7['exp'] = df7['exp'].apply(NaN_none).astype(str)

    df7['Description'] = (
        df7['question'] + ":\n" +
        df7['opa'] + "\n" +
        df7['opb'] + "\n" +
        df7['opc'] + "\n" +
        df7['opd'] + "\n" +
        "The subject is: " + df7['subject_name'] + "\n" +
        "and the topic is: " + df7['topic_name']
    )

    df7['Answer'] = (
        df7['cop'] + " choice. " +
        df7['exp']
    )

    df7['Description'] = df7['Description'].apply(clean_text)
    return df7[['Description', 'Answer']]

def load_and_process_data8():
    df8 = pd.read_csv('/content/medquad.csv')
    df8['Description'] = df8['question']
    df8['Answer'] = df8['answer'].astype(str)
    df8 = df8[['Description', 'Answer']]
    df8['Description'] = df8['Description'].apply(clean_text)
    df8['Answer'] = df8['Answer'].apply(clean_text)
    return df8

def combine_datasets():
    df1 = load_and_process_data1()
    df2 = load_and_process_data2()
    df3 = load_and_process_data3()
    df4 = load_and_process_data4()
    df5 = load_and_process_data5()
    df6 = load_and_process_data6()
    df7 = load_and_process_data7()
    df8 = load_and_process_data8()

    df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], ignore_index=True)
    return df

def save_to_csv(df, filename='data.csv'):
    df.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}")

def main():
    df = combine_datasets()
    return df

if __name__ == "__main__":
    df_data = main()
    save_to_csv(df_data)
    print("Dataset prepared for fine-tuning.")
