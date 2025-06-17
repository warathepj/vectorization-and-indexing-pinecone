# ซอร์สโค้ดนี้ ใช้สำหรับเป็นตัวอย่างเท่านั้น ถ้านำไปใช้งานจริง ผู้ใช้ต้องจัดการเรื่องความปลอดภัย และ ประสิทธิภาพด้วยตัวเอง

# Clinic Quotation RAG System

This repository contains a system for building a Retrieval Augmented Generation (RAG) application focused on clinic quotations. It leverages Pinecone for vector indexing and Google Gemini for embeddings, enabling efficient retrieval of relevant information from a knowledge base to augment language model responses.

## Project Structure

-   `vectorization_and_indexing.py`: A Python script responsible for processing markdown documents, generating vector embeddings using Google Gemini, and upserting them into a Pinecone index.
-   `site-content.md`: The primary markdown file containing the knowledge base content related to clinic quotations, which will be vectorized and indexed.
-   `.env.example`: An example file for setting up environment variables.
-   `.env`: (Not committed) Your actual environment variables for API keys.

## Features

-   **Pinecone Integration**: Utilizes Pinecone as a scalable vector database for efficient similarity search.
-   **Google Gemini Embeddings**: Generates high-quality vector representations of text using the Google Gemini `models/embedding-001`.
-   **Markdown Processing**: Handles markdown files, splitting them into manageable chunks for effective indexing.
-   **Retrieval Augmented Generation (RAG) Ready**: Provides the foundational components for building a RAG system to enhance language model capabilities with domain-specific knowledge.

## Setup and Installation

1.  **Clone the Repository**:

    ```bash
    git clone https://github.com/warathepj/vectorization-and-indexing-pinecone.git
    cd vectorization-and-indexing-pinecone
    ```

2.  **Environment Variables**:
    Create a `.env` file in the root directory and populate it with your API keys:

    ```
    PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
    PINECONE_ENVIRONMENT="YOUR_PINECONE_ENVIRONMENT"
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
    ```
3.  **Install Dependencies**:

    ```bash
    pip install -r requirements.txt # Assuming a requirements.txt will be created or use the direct install command below
    # OR
    pip install pinecone-client langchain_community langchain_google_genai tiktoken langchain_pinecone python-dotenv
    ```

## Usage

### 1. Prepare your Knowledge Base

Ensure your `site-content.md` file contains the relevant information you want to vectorize and index. This file serves as your knowledge base.

### 2. Vectorize and Index Data

Run the `vectorization_and_indexing.py` script to process `site-content.md`, generate embeddings, and upsert them into your Pinecone index:

```bash
python vectorization_and_indexing.py
```

This script will create a Pinecone index named `clinic-quotation-index` (if it doesn't exist) and populate it with your vectorized content.

### 3. Integrate with your RAG Application

Once the data is indexed in Pinecone, you can integrate this vector store into your RAG application. Use the Pinecone index to retrieve relevant document chunks based on user queries, and then pass these chunks to a large language model (e.g., Google Gemini) to generate informed responses.

## License

MIT License
