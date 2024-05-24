import os
import re
import json
import time
from typing import List, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationTokenBufferMemory
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from pydantic import BaseModel

chain = None


def initialize_chain(llm, vectorstore):
    # initialize the bm25 retriever and faiss retriever
    bm25_retriever = BM25Retriever.from_documents(all_splits)
    bm25_retriever.k = 2
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # initialize the ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
    )
    template = """
    Act as an intelligent assistant of Company Named LEAF.

    Context: {context}

    - Do not answer those question which is not related to the context.
    - Always aim to give reasoned, concise, and to-the-point answers.
    - Respond using correct grammar, proper English, and professional formatting.
    - Structure your answers based on the specific details or headings requested in the user's question.
    - If unsure about a particular topic or detail, admit your lack of knowledge rather than attempting to fabricate an answer.

    Chat History: {chat_history}


    Question: {question}
    Answer: """

    prompt = PromptTemplate(
        template=template, input_variables=["context", "chat history", "question"]
    )

    token_limit = 32000
    memory = ConversationTokenBufferMemory(
        llm=llm,
        max_token_limit=token_limit,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    global chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=ensemble_retriever,
        return_source_documents=True,
        chain_type="stuff",
        combine_docs_chain_kwargs={"prompt": prompt},
        memory=memory,
        verbose=False,
    )


class UserInputWithHistory(BaseModel):
    user_prompt: str
    chat_history: List[Tuple[str, str]] = []


def load(path):
    loaders = {".pdf": PyMuPDFLoader, ".csv": CSVLoader}

    def create_directory_loader(file_type, directory_path):
        return DirectoryLoader(
            path=directory_path,
            glob=f"**/*{file_type}",
            loader_cls=loaders[file_type],
        )

    documents = []
    new_pdf_loader = create_directory_loader(".pdf", path)
    new_csv_loader = create_directory_loader(".csv", path)

    documents.extend(new_pdf_loader.load())
    documents.extend(new_csv_loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    all_splits = text_splitter.split_documents(documents)
    return all_splits


all_splits = load("data/raw")
embeddings = OpenAIEmbeddings()

if len(all_splits) < 2000:
    vectorstore1 = FAISS.from_documents(all_splits, embeddings)
else:
    vectorstore1 = FAISS.from_documents(all_splits, embeddings)
    for i in range(2000, len(all_splits), 2000):
        time.sleep(5)
        end_index = min(i + 2000, len(all_splits))
        vectorstore2 = FAISS.from_documents(all_splits[i:end_index], embeddings)
        vectorstore1.merge_from(vectorstore2)

vectorstore1.save_local("data/vectorstores/faiss_index")
initialize_chain(
    ChatOpenAI(model_name="gpt-4-1106-preview", streaming=True), vectorstore1
)

chat_history = []


def response(query: str, chat_history: List[Tuple[str, str]]) -> str:
    result = chain(query)
    chat_history.append((query, result["answer"]))
    return result["answer"], chat_history


input_text = "Please enter your query, type exit to quit\nQuery: "
first_time = True
if __name__ == "__main__":
    while True:
        query = input(input_text if first_time else "Next Query: ")

        if not query:
            print("Please enter a valid query")
        else:
            if query == "exit":
                break

            response_text, updated_history = response(query, chat_history)
            chat_history.extend(updated_history)

            print(f"Answer: {response_text}")
            first_time = False
