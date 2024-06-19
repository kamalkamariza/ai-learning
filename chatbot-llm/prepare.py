import os
import json
import boto3
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv(".env")

LEAF_TABLE_NAME = os.getenv("LEAF_TABLE_NAME")
LEAF_DB_NAME = os.getenv("LEAF_DB_NAME")
LEAF_DB_USER = os.getenv("LEAF_DB_USER")
LEAF_DB_PASS = os.getenv("LEAF_DB_PASS")
LEAF_DB_HOST = os.getenv("LEAF_DB_HOST")
EFORM_TABLE_NAME = os.getenv("EFORM_TABLE_NAME")
EFORM_DB_NAME = os.getenv("EFORM_DB_NAME")
EFORM_DB_USER = os.getenv("EFORM_DB_USER")
EFORM_DB_PASS = os.getenv("EFORM_DB_PASS")
EFORM_DB_HOST = os.getenv("EFORM_DB_HOST")
S3_BUCKET = os.getenv("S3_BUCKET")
PREFIX = "files"
S3_CLIENT = boto3.client("s3")
MANAGEMENT_DIR = "./data/management"
USER_DIR = "./data/user"


def create_db_engine(user, password, host, db_name):
    return create_engine(f"mysql+mysqlconnector://{user}:{password}@{host}/{db_name}")


LEAF_ENGINE = create_db_engine(
    user=LEAF_DB_USER, password=LEAF_DB_PASS, host=LEAF_DB_HOST, db_name=LEAF_DB_NAME
)

EFORM_ENGINE = create_db_engine(
    user=EFORM_DB_USER,
    password=EFORM_DB_PASS,
    host=EFORM_DB_HOST,
    db_name=EFORM_DB_NAME,
)


def query_filepaths(query, engine):
    sql_query = pd.read_sql_query(
        query,
        engine,
    )
    return sql_query


def download_s3_files(queries):
    for _, query in queries.iterrows():
        filepath = query.get("filename")
        restricted = query.get("restricted")
        id_group = str(query.get("id_group"))
        target_folder = MANAGEMENT_DIR if restricted else USER_DIR
        input_filepath = os.path.join(PREFIX, filepath).replace("\\", "/")
        dst_folder = os.path.join(target_folder, id_group)
        output_filepath = os.path.join(dst_folder, filepath)
        if os.path.exists(output_filepath):
            continue
        os.makedirs(dst_folder, exist_ok=True)
        try:
            S3_CLIENT.download_file(S3_BUCKET, input_filepath, output_filepath)
        except Exception as e:
            print(f"Error downloading file {input_filepath}: {e}")


def save_csv_files(queries):
    for group_id, group_df in queries.groupby("id_group"):
        for _, row in group_df.iterrows():
            restricted = row["restricted"]
            form_title = row["form_title"].replace("/", "-")
            entry_date = int(row["entry_date"].timestamp())
            json_dict = json.loads(row.to_json())
            to_update = json.loads(json_dict.pop("data_components", "{}"))
            json_dict.update(to_update)
            dst_folder = MANAGEMENT_DIR
            if not restricted:
                dst_folder = USER_DIR

            filename = f"{form_title}_{group_id}_{entry_date}.json"
            full_path = os.path.join(dst_folder, str(group_id), filename)

            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            with open(full_path, "w") as f:
                json.dump(json_dict, f, indent=4)


if __name__ == "__main__":
    # leaf_queries = query_filepaths(
    #     query=f"""
    # SELECT id_group, restricted, filename
    # FROM {LEAF_TABLE_NAME}
    # """,
    #     engine=LEAF_ENGINE,
    # )
    # download_s3_files(leaf_queries)

    eform_queries = query_filepaths(
        query=f"""
    SELECT id_group, data_components, restricted, form_title, entry_date
    FROM {EFORM_TABLE_NAME}
    WHERE entry_date > '2024-05-10' AND deleted=0
    """,
        engine=EFORM_ENGINE,
    )
    save_csv_files(eform_queries)
