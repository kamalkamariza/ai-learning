import os
import time
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever

gpt_model_1 = "gpt-4-1106-preview"
gpt_model_2 = "gpt-3.5-turbo-0125"
store = {}
session_id = "test"
input_text = "Please enter your query, type exit to quit\nQuery: "
first_time = True
data_path = "data/raw"


def load_retriever(path):
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
    print("documents", documents)
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


def get_session_history(session_id) -> BaseChatMessageHistory:
    global store
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def load_agent(llm, tools, prompt):
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    return agent_with_chat_history


if __name__ == "__main__":
    tools = load_retriever(data_path)
    print(tools)
    llm = ChatOpenAI(model=gpt_model_2, temperature=0.1)
    prompt = load_prompt()
    agent_with_chat_history = load_agent(llm=llm, tools=tools, prompt=prompt)

    while True:
        query = input(input_text if first_time else "Next Query: ")

        if not query:
            print("Please enter a valid query")
        else:
            if query == "exit":
                break

            response = agent_with_chat_history.invoke(
                {"input": query}, config={"configurable": {"session_id": session_id}}
            )
            response_text = response["output"]

            print(f"Answer: {response_text}")
            first_time = False
