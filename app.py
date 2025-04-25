# ─── Imports ─────────────────────────────────────────────────────────────────
import os
import sys
import getpass
from decimal import Decimal
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

from typing_extensions import TypedDict, Annotated

from huggingface_hub import login
from sympy.physics.units import temperature  # Unused? Remove if unnecessary

from langchain_community.utilities import SQLDatabase
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool

from langchain.prompts import FewShotPromptTemplate, SemanticSimilarityExampleSelector
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import init_chat_model
from langgraph.graph import START, StateGraph

from few_shots import examples


# ─── Environment Setup ──────────────────────────────────────────────────────
env_path = Path('C:/Users/Assia/PycharmProjects/PythonProject/env')
load_dotenv(dotenv_path=env_path)

# ─── Database Initialization ────────────────────────────────────────────────

#db info
db_user = "root"
db_password = os.getenv('DB_PASSWORD')
db_host = "localhost"
db_name = "atliq_tshirts"

# create the  SQL db object
db = SQLDatabase.from_uri(
        f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",
       sample_rows_in_table_info=3
     )

# ─── HuggingFace Embeddings ─────────────────────────────────────────────────
login(os.getenv('HUGGING_FACE_KEY'))  # HF login
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ─── Mistral AI Setup ───────────────────────────────────────────────────────
if not os.environ.get("MISTRAL_API_KEY"):
    os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter API key for Mistral AI: ")

llm = init_chat_model("mistral-large-latest", model_provider="mistralai")


# ─── Few-Shot Examples & Vectorstore ────────────────────────────────────────
to_vectorize = [" ".join(example.values()) for example in examples]

vectorstore = Chroma.from_texts(
    texts=to_vectorize,
    embedding=embeddings,
    metadatas=examples
)

example_selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=5,
    input_keys=["input"],
)

example_prompt = PromptTemplate.from_template("User input: {input}\nSQL query: {query}")

prefix = """
You are a MySQL expert. Given an input question, create a syntactically correct MySQL query...
(Shortened for clarity)
"""

# ─── LangGraph State Definition ─────────────────────────────────────────────
class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str


class QueryOutput(TypedDict):
    query: Annotated[str, ..., "Syntactically valid SQL query."]


# ─── LangGraph Chain Steps ──────────────────────────────────────────────────
def write_query(state: State):
    prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix="User input: {input}\nSQL query: ",
        input_variables=["input", "table_info", "top_k"],
    ).format(input=state['question'], table_info=db.table_info, top_k=3)

    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)

    return {"question": state["question"], "query": result["query"]}


def execute_query(state: State):
    tool = QuerySQLDatabaseTool(db=db)
    return {
        "question": state["question"],
        "query": state["query"],
        "result": tool.invoke(state["query"])
    }


def generate_answer(state: State):
    if not state["result"]:
        return {**state, "answer": "No result found!"}

    prompt = (
        f"Given the following user question, SQL query, and result, answer the question:\n\n"
        f"Question: {state['question']}\n"
        f"SQL Query: {state['query']}\n"
        f"SQL Result: {state['result']}"
    )
    response = llm.invoke(prompt)
    return {**state, "answer": response.content}


# ─── LangGraph Compilation ──────────────────────────────────────────────────
graph = StateGraph(State)\
    .add_sequence([write_query, execute_query, generate_answer])\
    .add_edge(START, "write_query")\
    .compile()


# ─── Streamlit Interface ────────────────────────────────────────────────────
st.set_page_config(page_title="NexQuery", page_icon="🧐")
st.title("🧐 NexQuery: Ask Your Database Anything!")

question = st.text_input("💬 Ask a question to interact with the database:")

if question:
    with st.status("⏳ Processing your question..."):
        answer_dict = graph.invoke(input={"question": question})

    st.success("✅ Answer received!")
    st.markdown("### 💡 Answer")
    st.write(answer_dict['answer'])

    with st.expander("🛠 SQL Query"):
        st.code(answer_dict['query'], language="sql")

    with st.expander("📊 Raw SQL Result"):
        st.write(answer_dict['result'])
