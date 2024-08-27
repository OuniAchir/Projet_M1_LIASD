import pandas as pd
import pyarrow as pa
from datasets import Dataset
from scrapping_PMC import main as load_pmc
#from Scrapping_ArXiv import main as load_arxiv
from Load_Data import main as load_data

def load_pmc_data():
    return load_pmc()

#def load_arxiv_data():
    #return load_arxiv()

def load__data():
    return load_data()

def combine_all_data():
    df_pmc = load_pmc_data()
    #df_arxiv = load_arxiv_data()
    df_load = load__data()
    combined_df = pd.concat([df_pmc, df_load], ignore_index=True)
    return combined_df

def save_to_csv(df, filename='combined_data.csv'):
    df.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}")

def create_arrow_dataset(df):
    table = pa.Table.from_pandas(df)
    dataset = Dataset(table)
    return dataset

if __name__ == "__main__":
    combined_df = combine_all_data()
    save_to_csv(combined_df)
    dataset = create_arrow_dataset(combined_df)
    print("Combined dataset prepared and saved.")
