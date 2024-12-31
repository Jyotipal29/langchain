# from dotenv import load_dotenv


# load_dotenv()


# from  langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import ReadTheDocsLoader 
# # from langchain_openai import OpenAIEmbeddings
# from langchain.embeddings.openai import OpenAIEmbeddings

# # from langchain_pinecone import PineconeVectorStore
# from langchain.vectorstores import Pinecone
# from pinecone import Pinecone, ServerlessSpec
# from langchain_community.vectorstores import Pinecone

# import os
# from langchain.schema import Document
# from langchain_community.document_loaders.base import BaseDocumentLoader
# # Create a Pinecone client instance
# pinecone_client = Pinecone(
#     api_key=os.getenv("PINECONE_API_KEY")
# )


# # Ensure the index exists
# index_name = "langchain-doc-index"
# if index_name not in pinecone_client.list_indexes().names():
#     pinecone_client.create_index(
#         name=index_name,
#         dimension=1536,  # Adjust to match the dimension of your embeddings
#         metric='cosine',  # Change metric as needed ('cosine', 'euclidean', etc.)
#         spec=ServerlessSpec(
#             cloud='aws',
#             region=os.getenv("PINECONE_ENVIRONMENT")
#         )
#     )

# # Define embeddings
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# class UTF8ReadTheDocsLoader(BaseDocumentLoader):
#     def __init__(self, path: str):
#         self.path = path

#     def load(self):
#         """Load documents with enforced UTF-8 encoding."""
#         documents = []
#         for root, _, files in os.walk(self.path):
#             for file_name in files:
#                 file_path = os.path.join(root, file_name)
#                 try:
#                     with open(file_path, 'r', encoding='utf-8') as file:
#                         content = file.read()
#                         # Add the document with metadata
#                         documents.append(Document(page_content=content, metadata={"source": file_path}))
#                 except Exception as e:
#                     print(f"Error reading file {file_path}: {e}")
#         return documents
    


# def ingest_docs():
#      # loader = ReadTheDocsLoader("langchain-docs/api.python.langchain.com/en/latest")
#      loader =UTF8ReadTheDocsLoader("langchain-docs/api.python.langchain.com/en/latest")
#      raw_documents = loader.load()
#      print(f"loaded  {len(raw_documents)} documents")

#      text_splitter = RecursiveCharacterTextSplitter(chunk_size=600,chunk_overlap=50)
#      documents = text_splitter.split_documents(raw_documents)
#      for doc in documents:
#           new_url = doc.metadata["source"]
#           new_url = new_url.replace("langchain-docs","https:/")
#           doc.metadata.update({"source":new_url})
#      print(f"going to add {len(documents)} to Pinecone")
#      # Pinecone.from_documents(
#      #      documents,embeddings,index_name="langchain-doc-index"
#      # )
#      LangchainPinecone.from_documents(
#         documents, embeddings, index_name=index_name
#      )
#      print("*** Loading the vectorestore done   ***")




# if __name__ == "__main__":
#     ingest_docs()





from dotenv import load_dotenv
import os
from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_community.vectorstores import Pinecone

# Load environment variables
load_dotenv()

# Pinecone Initialization
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
index_name = "langchain-doc-index"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
class UTF8ReadTheDocsLoader:
    def __init__(self, path: str):
        self.path = path

    def load(self):
        """Loads documents with enforced UTF-8 encoding."""
        documents = []
        for root, _, files in os.walk(self.path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                try:
                    with open(file_path, mode="r", encoding="utf-8") as file:
                        content = file.read()
                        # Add the document with metadata
                        documents.append(Document(page_content=content, metadata={"source": file_path}))
                except UnicodeDecodeError as e:
                    print(f"Error reading file {file_path}: {e}")
        return documents
    

def ingest_docs():
#     loader = ReadTheDocsLoader("langchain-docs/api.python.langchain.com/en/latest")
    loader = UTF8ReadTheDocsLoader("langchain-docs/api.python.langchain.com/en/latest")
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)

    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to add {len(documents)} to Pinecone")
    Pinecone.from_documents(documents, embeddings, index_name=index_name)
    print("*** Loading the vectorstore done ***")

if __name__ == "__main__":
    ingest_docs()
