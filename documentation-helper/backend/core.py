# from dotenv import load_dotenv
# from langchain.chains.retrieval import create_retrieval_chain

# # Load environment variables
# load_dotenv()

# from langchain import hub 
# from langchain.chains.combine_documents import create_stuff_documents_chain 
# from langchain_pinecone import Pinecone
# # from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain.chat_models import ChatOpenAI
# from langchain.embeddings import OpenAIEmbeddings
# import pinecone
# import os
# INDEX_NAME = "langchain-doc-index"
# pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))
# def run_llm(query:str):
#     embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
#     docsearch = Pinecone.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)
#     chat = ChatOpenAI(verbose=True,temperature=0)
#     retrieval_qa_chat_prompt= hub.pull("langchain-ai/retrieval-qa-chat")
#     stuff_documents_chain = create_stuff_documents_chain(chat,retrieval_qa_chat_prompt)
#     qa = create_retrieval_chain(retriever=docsearch.as_retriever(),combine_docs_chain=stuff_documents_chain)
#     result = qa.invoke(input={"input":query})
#     return result


# if __name__ == "__main__":
#     res = run_llm(query="what is a langchain chain ?")
#     print(res["answer"])





from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import Pinecone
from langchain import hub
from pinecone import Pinecone as PineconeClient, ServerlessSpec
import os

# Load environment variables
load_dotenv()

INDEX_NAME = "langchain-doc-index"

# Initialize Pinecone
pinecone_client = PineconeClient(
    api_key=os.getenv("PINECONE_API_KEY"),
)

# Ensure the index exists
if INDEX_NAME not in pinecone_client.list_indexes().names():
    pinecone_client.create_index(
        name=INDEX_NAME,
        dimension=1536,  # Match this with the dimension of your embeddings
        metric='cosine',  # Use 'euclidean', 'cosine', or 'dotproduct' as needed
        spec=ServerlessSpec(
            cloud='aws',
            region=os.getenv("PINECONE_ENVIRONMENT"),
        )
    )

def run_llm(query: str):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docsearch = Pinecone.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)
    chat = ChatOpenAI(verbose=True, temperature=0)
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)
    qa = create_retrieval_chain(retriever=docsearch.as_retriever(), combine_docs_chain=stuff_documents_chain)
    result = qa.invoke(input={"input": query})
    return result

if __name__ == "__main__":
    res = run_llm(query="what is a langchain chain ?")
    print(res["answer"])
