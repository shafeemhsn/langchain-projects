import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter
from langchain_unstructured import UnstructuredLoader


load_dotenv()


if __name__ == '__main__':
    print("Ingesting...")

    loader = UnstructuredLoader(
        file_path=Path(__file__).with_name("mediumblog1.txt"),
        chunking_strategy="basic",
        max_character=1000000,
    )
    document = loader.load()

    print(f"Loaded {len(document)} document elements")

    print("Splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))


    print("Ingesting to pinecone...")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ['INDEX_NAME'])
    print("finish")
