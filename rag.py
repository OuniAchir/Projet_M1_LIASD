import logging
import sys
from llama_index import SimpleDirectoryReader, KnowledgeGraphIndex, Settings
from llama_index.graph_stores import SimpleGraphStore
from llama_index import StorageContext
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from pyvis.network import Network
from transformers import pipeline
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from transformers import HuggingFaceInferenceAPI

def setup_logging():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

def load_documents(directory_path):
    documents = SimpleDirectoryReader(directory_path).load_data()
    print(f"Loaded {len(documents)} documents.")
    return documents

def setup_llm(model_name, token):
    llm = HuggingFaceInferenceAPI(model_name=model_name, token=token)
    return llm

def setup_embeddings():
    embed_model = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="thenlper/gte-large")
    )
    return embed_model

def setup_storage_context():
    graph_store = SimpleGraphStore()
    storage_context = StorageContext.from_defaults(graph_store=graph_store)
    return storage_context

def construct_knowledge_graph(documents, embed_model, storage_context):
    index = KnowledgeGraphIndex.from_documents(
        documents=documents,
        max_triplets_per_chunk=3,
        storage_context=storage_context,
        embed_model=embed_model,
        include_embeddings=True
    )
    return index

def query_knowledge_graph(index, query):
    response = index.query(query)
    return response

def main():
    setup_logging()

    # Charger les documents
    directory_path = "/content/data"
    documents = load_documents(directory_path)

    # Configurer le LLM
    model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    secret_hf = "your_hugging_face_token"  # Remplacez par votre token Hugging Face
    llm = setup_llm(model_name, secret_hf)

    # Configurer les embeddings
    embed_model = setup_embeddings()

    # Configurer le contexte de stockage
    storage_context = setup_storage_context()

    # Construire le Knowledge Graph Index
    index = construct_knowledge_graph(documents, embed_model, storage_context)

    # Poser une question pour tester la génération de réponse
    query = "Comment les politiques de sécurité numérique peuvent-elles être harmonisées entre les pays de l'OCDE pour réduire les incidents de cybersécurité tout en protégeant la vie privée des citoyens ?"
    response = query_knowledge_graph(index, query)

    # Afficher la réponse
    print(response)

if __name__ == "__main__":
    main()
