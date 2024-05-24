import os
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.agents.agent_types import AgentType
from langchain_community.vectorstores import FAISS
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.tools.retriever import create_retriever_tool
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings
from fastapi import FastAPI, UploadFile, HTTPException, File, Form
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders import PyMuPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain.agents import (
    AgentExecutor,
    create_tool_calling_agent,
    initialize_agent,
    Tool,
)

GPT_4 = "gpt-4-1106-preview"
GPT_3 = "gpt-3.5-turbo-0125"
CSV_FOLDER = "../data/raw/csv"
PDF_FOLDER = "../data/raw/pdf"
LLM = ChatOpenAI(temperature=0, model=GPT_4)
SESSION_ID = "test"
STORE = {}

app = FastAPI()


def load_csv_agent():
    csv_files = [os.path.join(CSV_FOLDER, path) for path in os.listdir(path=CSV_FOLDER)]
    agent = create_csv_agent(
        LLM,
        csv_files,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )
    return agent


def load_doc_agent():
    tools = []
    document_pages = []
    files = [
        {
            "name": "LEAF  Services Agreement- 20181001",
            "path": "./data/raw/pdf/LEAF  Services Agreement- 20181001.pdf",
        },
        {
            "name": "LEAF Customer Agreement 20170625",
            "path": "./data/raw/pdf/LEAF Customer Agreement 20170625.pdf",
        },
        {
            "name": "MK11 Facilities",
            "path": "./data/raw/pdf/MK11 Facilities.pdf",
        },
        {
            "name": "MK11 House Rules",
            "path": "./data/raw/pdf/MK11 House Rules.pdf",
        },
    ]

    for file in files:
        loader = PyPDFLoader(file["path"])
        pages = loader.load_and_split()
        document_pages.extend(pages)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(document_pages)
    embeddings = OpenAIEmbeddings()
    retriever = FAISS.from_documents(docs, embeddings).as_retriever()

    # Wrap retrievers in a Tool
    tools.append(
        Tool(
            # args_schema=DocumentInput,
            name="doc_retriever",
            description=f"useful when you want to answer questions about non csv files or non analysis",
            func=RetrievalQA.from_chain_type(llm=LLM, retriever=retriever),
        )
    )

    agent = initialize_agent(
        agent=AgentType.OPENAI_FUNCTIONS,
        tools=tools,
        llm=LLM,
        verbose=True,
    )

    return agent


def load_pdf_agent():
    def load_retriever(path):
        loaders = {
            ".pdf": PyMuPDFLoader,
        }

        def create_directory_loader(file_type, directory_path):
            return DirectoryLoader(
                path=directory_path,
                glob=f"**/*{file_type}",
                loader_cls=loaders[file_type],
            )

        documents = []
        loader = create_directory_loader(".pdf", path)

        documents.extend(loader.load())
        documents = RecursiveCharacterTextSplitter(
            chunk_size=1024, chunk_overlap=100
        ).split_documents(documents)
        vector = FAISS.from_documents(documents, OpenAIEmbeddings())
        vector.save_local("data/vectorstores/faiss_index_agent")

        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 2
        faiss_retriever = vector.as_retriever(search_kwargs={"k": 2})
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
        )
        retriever_tool = create_retriever_tool(
            ensemble_retriever,
            "bm25_faiss",
            "Search againts vector store based on text",
        )

        return [retriever_tool]

    def load_prompt():
        system_prompt = """
            Act as an intelligent assistant of Company Named LEAF.
            Do not answer those question which is not related to the context.
            Always aim to give reasoned, concise, and to-the-point answers.
            Respond using correct grammar, proper English, and professional formatting.
            Structure your answers based on the specific details or headings requested in the user's question.
            If unsure about a particular topic or detail, admit your lack of knowledge rather than attempting to fabricate an answer.
            """

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="chat_history"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ],
        )

        return prompt

    tools = load_retriever(PDF_FOLDER)
    agent = create_tool_calling_agent(LLM, tools, load_prompt())
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor


def load_main_agent():
    csv_files = [os.path.join(CSV_FOLDER, path) for path in os.listdir(path=CSV_FOLDER)]

    tools = []
    document_pages = []
    files = [
        {
            "name": "LEAF  Services Agreement- 20181001",
            "path": f"{PDF_FOLDER}/LEAF  Services Agreement- 20181001.pdf",
        },
        {
            "name": "LEAF Customer Agreement 20170625",
            "path": f"{PDF_FOLDER}/LEAF Customer Agreement 20170625.pdf",
        },
        {
            "name": "MK11 Facilities",
            "path": f"{PDF_FOLDER}/MK11 Facilities.pdf",
        },
        {
            "name": "MK11 House Rules",
            "path": f"{PDF_FOLDER}/MK11 House Rules.pdf",
        },
    ]

    for file in files:
        loader = PyPDFLoader(file["path"])
        pages = loader.load_and_split()
        document_pages.extend(pages)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(document_pages)
    embeddings = OpenAIEmbeddings()
    retriever = FAISS.from_documents(docs, embeddings).as_retriever()

    # Wrap retrievers in a Tool
    tools.append(
        Tool(
            # args_schema=DocumentInput,
            name="doc_retriever",
            description=f"useful when you want to answer questions about non csv files or non analysis",
            func=RetrievalQA.from_chain_type(llm=LLM, retriever=retriever),
        )
    )

    agent = create_csv_agent(
        LLM,
        csv_files,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        extra_tools=tools,
    )
    return agent


def get_session_history(session_id) -> BaseChatMessageHistory:
    global STORE
    if session_id not in STORE:
        STORE[session_id] = ChatMessageHistory()
    return STORE[session_id]


def init_chatbot():
    agent_executor = load_main_agent()
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    return agent_with_chat_history


agent_with_chat_history = init_chatbot()


@app.get("/response")
async def get_response(prompt: str):
    response = agent_with_chat_history.invoke(
        {"input": prompt}, config={"configurable": {"session_id": SESSION_ID}}
    )
    return response


if __name__ == "__main__":
    # agent_executor = load_csv_agent()
    # agent_executor = load_pdf_agent()
    # agent_executor = load_doc_agent()
    agent_executor = load_main_agent()
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    while True:
        prompt = input("Please input your query: ")
        agent_with_chat_history.invoke(
            {"input": prompt}, config={"configurable": {"session_id": SESSION_ID}}
        )
