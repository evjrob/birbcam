import sqlite3
import os
import click

DB_PATH = os.getenv("DB_PATH", "../data/model_results.db")
DATA_DIR = os.getenv("DATA_DIR", "../data/")

def create_db():
    con = sqlite3.connect(DB_PATH)
    cursor = con.cursor()
    sql_file_path = os.path.join(DATA_DIR, "db_schema.sql")
    sql_schema = open(sql_file_path)
    sql_as_string = sql_schema.read()
    print(sql_as_string)
    cursor.executescript(sql_as_string)

@click.command()
@click.argument("utility")
def run_script(utility):
    if utility == "create_db":
        create_db()
    else:
        print("Did not understand argument")

run_script()
