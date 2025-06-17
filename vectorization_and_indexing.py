pip install pinecone langchain_community langchain_google_genai tiktoken langchain_pinecone

import os
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from uuid import uuid4

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

try:
    from dotenv import load_dotenv
    load_dotenv()
    print(".env file loaded successfully.")
except ImportError:
    print("python-dotenv not installed. Please install it using: !pip install python-dotenv")
except Exception as e:
    print(f"Error loading .env file: {e}")

import os

pinecone_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
google_key = os.getenv("GOOGLE_API_KEY")

print(f"PINECONE_API_KEY: {pinecone_key}")
print(f"PINECONE_ENVIRONMENT: {pinecone_env}")
print(f"GOOGLE_API_KEY: {google_key}")

if not pinecone_key or not pinecone_env or not google_key:
    print("One or more environment variables are still not set.")
else:
    print("All environment variables are set.")

if not os.getenv("PINECONE_API_KEY") or not os.getenv("PINECONE_ENVIRONMENT") or not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("Please set PINECONE_API_KEY, PINECONE_ENVIRONMENT, and GOOGLE_API_KEY environment variables.")
else:
    print("All environment variables are set.")

INDEX_NAME = "clinic-quotation-index"
MD_FILE_PATH = "site-content.md"
EMBEDDING_MODEL_NAME = "models/embedding-001"
GEMINI_EMBEDDING_DIMENSION = 768

print(f"Initializing Pinecone client...")
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

if INDEX_NAME not in pc.list_indexes():
    print(f"Creating Pinecone index: {INDEX_NAME}...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=GEMINI_EMBEDDING_DIMENSION, # Changed dimension for Gemini
        metric="cosine",  # Cosine similarity is common for embeddings
        spec=ServerlessSpec(cloud='aws', region='us-east-1') # Or choose your desired cloud/region
    )
    print(f"Index {INDEX_NAME} created successfully.")
else:
    print(f"Connecting to existing Pinecone index: {INDEX_NAME}...")

index = pc.Index(INDEX_NAME)

print(f"Loading and processing Markdown file: {MD_FILE_PATH}...")

with open(MD_FILE_PATH, 'r', encoding='utf-8') as f:
    markdown_document = f.read()

headers_to_split_on = [
    ("#", "Header1"),
    ("##", "Header2"),
    ("###", "Header3"),
]
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)
md_header_splits = markdown_splitter.split_text(markdown_document)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, # Adjust as needed
    chunk_overlap=50, # Adjust as needed
    length_function=len,
)
final_chunks = text_splitter.split_documents(md_header_splits)

print(f"Split {MD_FILE_PATH} into {len(final_chunks)} chunks.")

print(f"Initializing Google Gemini embeddings with model: {EMBEDDING_MODEL_NAME}...")
embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME, google_api_key=GOOGLE_API_KEY)

print("Upserting chunks to Pinecone...")

PineconeVectorStore.from_documents(
    final_chunks,
    embeddings,
    index_name=INDEX_NAME
)

print(f"Successfully inserted data from {MD_FILE_PATH} into Pinecone index '{INDEX_NAME}'.")

print("\nVerifying data in Pinecone (fetching index stats)...")
index_stats = index.describe_index_stats()
print(index_stats)