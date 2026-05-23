import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
finance_retriever = vectorstore.as_retriever(search_kwargs={"k": 4, "filter": {"domain": "finance"}})

docs = finance_retriever.invoke("What was Acme Corp's total revenue in Q3?")
print("FOUND DOCUMENTS:", len(docs))
for d in docs:
    print(d.page_content)
    print("-----")
