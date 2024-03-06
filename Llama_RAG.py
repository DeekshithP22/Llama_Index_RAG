import logging
import sys

from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def setup_logging():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

def setup_llm(api_key, azure_endpoint, api_version):
    llm = AzureOpenAI(
        model="gpt-35-turbo-16k",
        deployment_name="my-custom-llm",
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        api_version=api_version,
    )
    return llm

def setup_embedding(api_key, azure_endpoint, api_version):
    embed_model = AzureOpenAIEmbedding(
        model="text-embedding-ada-002",
        deployment_name="my-custom-embedding",
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        api_version=api_version,
    )
    return embed_model

def setup_index(documents):
    index = VectorStoreIndex.from_documents(documents)
    return index

def setup_huggingface_embedding():
    Settings.embed_model = HuggingFaceEmbedding(model_name='BAAI/bge-small-en-v1.5')

def main():
    setup_logging()

    api_key = "<api-key>"
    azure_endpoint = "https://<your-resource-name>.openai.azure.com/"
    api_version = "2023-07-01-preview"

    llm = setup_llm(api_key, azure_endpoint, api_version)
    embed_model = setup_embedding(api_key, azure_endpoint, api_version)
    Settings.llm = llm
    Settings.embed_model = embed_model

    documents = SimpleDirectoryReader(input_files=["../../data/paul_graham/paul_graham_essay.txt"]).load_data()

    index = setup_index(documents)

    setup_huggingface_embedding()

    query = "What is most interesting about this essay?"
    query_engine = index.as_query_engine()
    answer = query_engine.query(query)

    print(answer.get_formatted_sources())
    print("query was:", query)
    print("answer was:", answer)

if __name__ == "__main__":
    main()
