import os
import json
import pandas as pd
from glob import glob
from tqdm import tqdm
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.agents.agent_types import AgentType
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_experimental.agents.agent_toolkits import (
    create_csv_agent,
    create_pandas_dataframe_agent,
)
from langchain.agents import AgentExecutor, create_tool_calling_agent, Tool
from langchain_community.document_loaders.csv_loader import UnstructuredCSVLoader

load_dotenv()

GPT_4 = "gpt-4-1106-preview"
GPT_3 = "gpt-3.5-turbo-0125"
MANAGEMENT_DIR = "./data/management"
USER_DIR = "./data/user"
LLM = ChatOpenAI(temperature=0, model=GPT_4)
TOOLS = []
SESSION_ID = "test"
USER_ROLE = "management"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 0
STORE = {}


def load_files(dirs, ext):
    all_files = []
    for dir in dirs:
        pattern = os.path.join(dir, "**", f"*{ext}")
        files = tqdm(
            glob(pattern, recursive=True), desc=f"Loading {ext} files for {dir}"
        )
        all_files.extend(files)
    return all_files


def init_df_agent(user):
    dfs = []
    if user in ["management"]:
        print("df creating for management")
        files = load_files([MANAGEMENT_DIR, USER_DIR], ".xlsx")
    else:
        print("df creating for user")
        files = load_files([USER_DIR], ".xlsx")

    for file in files:
        df = pd.read_excel(file)
        dfs.append(df)

    agent = create_pandas_dataframe_agent(
        llm=LLM, df=dfs, agent_type=AgentType.OPENAI_FUNCTIONS, verbose=True
    )
    
    return agent


def init_pdf_retriever(user):
    document_pages = []
    if user in ["management"]:
        print("pdf creating for management")
        files = load_files([MANAGEMENT_DIR, USER_DIR], ".pdf")
        vectorstore_path = "data/vectorstores/admin_retriever"
    else:
        print("pdf creating for user")
        files = load_files([USER_DIR], ".pdf")
        vectorstore_path = "data/vectorstores/user_retriever"

    for file in files:
        loader = PyPDFLoader(file)
        pages = loader.load_and_split()
        document_pages.extend(pages)

    text_splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    docs = text_splitter.split_documents(document_pages)
    embeddings = OpenAIEmbeddings()
    vector = FAISS.from_documents(docs, embeddings)
    vector.save_local(vectorstore_path)
    retriever = vector.as_retriever()
    return retriever


def init_csv_agent(user):
    if user in ["management"]:
        print("csv creating for management")
        files = load_files([MANAGEMENT_DIR, USER_DIR], ".csv")
    else:
        print("csv creating for user")
        files = load_files([USER_DIR], ".csv")

    agent = create_csv_agent(
        LLM, files, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS
    )

    return agent


TOOLS.extend(
    [
        Tool(
            name="admin_doc_retriever",
            description=f"useful when you want to answer admin or management asked questions about non csv files or non analysis",
            func=RetrievalQA.from_chain_type(
                llm=LLM, retriever=init_pdf_retriever("management")
            ),
        ),
        Tool(
            name="user_doc_retriever",
            description=f"useful when you want to answer user asked questions about non csv files or non analysis",
            func=RetrievalQA.from_chain_type(
                llm=LLM, retriever=init_pdf_retriever("user")
            ),
        ),
        Tool.from_function(
            func=init_df_agent("management").run,
            name="admin_df_retriever",
            description=f"useful when you want to answer admin or management asked questions that require analysis or dataframe to calculate on",
        ),
        # Tool.from_function(
        #     func=init_csv_agent("management").run,
        #     name="admin_csv_retriever",
        #     description=f"useful when you want to answer admin or management asked questions that require csv files or analysis",
        # ),
        # Tool.from_function(
        #     func=init_csv_agent("user").run,
        #     name="user_csv_retriever",
        #     description=f"useful when you want to answer user asked questions that require csv files or analysis",
        # ),
    ]
)


def get_session_history(session_id) -> BaseChatMessageHistory:
    global STORE
    if session_id not in STORE:
        STORE[session_id] = ChatMessageHistory()
    return STORE[session_id]


def load_prompt():
    # system_prompt = """
    #     Act as an intelligent assistant of Company Named LEAF.
    #     Do not answer those question which is not related to the context.
    #     Always aim to give reasoned, concise, and to-the-point answers.
    #     Respond using correct grammar, proper English, and professional formatting.
    #     Structure your answers based on the specific details or headings requested in the user's question.
    #     If unsure about a particular topic or detail, admit your lack of knowledge rather than attempting to fabricate an answer.
    #     Do not give a general answer or not related to the context of this company.
    #     Use specific tools related to their access level of documents such as use admin tools for admin accessed knowledge and use user tools for user accessed knowledge.
    #     """

    system_prompt = """
    You're a LEAF smart chatbot assistant, you will answer questions related to the LEAF context. Do not give general feedback. if a response cannot be formed strictly using the context, politely say you don't have enough knowledge to answer.
    """

    # system_prompt = """
    # Act as an intelligent assistant of Company Named LEAF.

    # - Always aim to give reasoned, concise, and to-the-point answers.
    # - Respond using correct grammar, proper English, and professional formatting.
    # - Structure your answers based on the specific details or headings requested in the user's question.
    # - If unsure about a particular topic or detail, admit your lack of knowledge rather than attempting to fabricate an answer.
    
    # You can only make conversations based on the provided context. If a response cannot be formed strictly using the context, politely say you don’t have knowledge about that topic.

    # Chat History: {chat_history}

    # Question: {input}
    # Answer: 
    # """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="chat_history"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ],
    )

    return prompt


agent = create_tool_calling_agent(LLM, TOOLS, load_prompt())
agent_executor = AgentExecutor(
    agent=agent, 
    tools=TOOLS,
    return_intermediate_steps=True,
    verbose=True
)
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

while True:
    USER_ROLE = input("Please input your user role: ")
    prompt = input("Please input your query: ")
    role_prompt = "As a management, " if USER_ROLE == "management" else "As a user, "
    prompt = f"{role_prompt} {prompt}"
    print(prompt)
    response = agent_with_chat_history.invoke(
        {"input": prompt}, config={"configurable": {"session_id": SESSION_ID}}
    )
    print(STORE)
    print(response["output"].encode("utf-8"))
