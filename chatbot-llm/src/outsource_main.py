import os
import pkgutil
import tiktoken
from glob import glob
from tqdm import tqdm
from pymongo import MongoClient
from longtrainer.trainer import LongTrainer
from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader

PROMPT = """
    Act as an intelligent assistant of Company Named LEAF.

    Context: {context}

    - Always aim to give reasoned, concise, and to-the-point answers.
    - Respond using correct grammar, proper English, and professional formatting.
    - Structure your answers based on the specific details or headings requested in the user's question.
    - If unsure about a particular topic or detail, admit your lack of knowledge rather than attempting to fabricate an answer.

    Chat History: {chat_history}


    Question: {question}
    Answer: 
    """

PDF_DIR = "./data/raw/pdf"
CSV_DIR = "./data/raw/csv"

mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["chatbot_db"]  # Database name
bots_collection = db["bots"]


def load_documents(file_types=["pdf"]):
    documents = []
    if "pdf" in file_types:
        pdf_pattern = os.path.join(PDF_DIR, "**", "*.pdf")
        for pdf_path in tqdm(
            glob(pdf_pattern, recursive=True), desc="Loading Documents"
        ):
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())

    if "csv" in file_types:
        csv_pattern = os.path.join(CSV_DIR, "**", "*.csv")
        for csv_path in tqdm(
            glob(csv_pattern, recursive=True), desc="Loading Documents"
        ):
            loader = CSVLoader(file_path=csv_path)
            documents.extend(loader.load())

    return documents


def configure_trainer(file_types=["pdf"]):
    trainer = LongTrainer(
        mongo_endpoint="mongodb://localhost:27017/", prompt_template=PROMPT, num_k=3
    )
    bot_id = trainer.initialize_bot_id()
    # Store bot_id in MongoDB
    bot_document = {"bot_id": bot_id}
    bots_collection.insert_one(bot_document)
    document = load_documents(file_types)
    trainer.pass_documents(document, bot_id)
    trainer.create_bot(bot_id)
    return trainer, bot_id


def new_chat(bot_id):
    # Start a New Chat
    chat_id = trainer.new_chat(bot_id)
    bots_collection.update_one({"bot_id": bot_id}, {"$push": {"chat_ids": chat_id}})
    return chat_id


def num_tokens(string, encoding_name):
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


if __name__ == "__main__":
    trainer, bot_id = configure_trainer(["csv", "pdf"])
    chat_id = new_chat(bot_id)
    input_text = "Please enter your query, type exit to quit\nQuery: "
    first_time = True
    while True:
        query = input(input_text if first_time else "Next Query: ")

        if not query:
            print("Please enter a valid query")
        else:
            if query == "exit":
                break

            token = num_tokens(query, "gpt-4-1106-preview")
            if token <= 16000:
                response, _ = trainer.get_response(query, bot_id, chat_id)
            else:
                response = "Please reduce the input context for the query."

            print(f"Answer: {response}")
            first_time = False
