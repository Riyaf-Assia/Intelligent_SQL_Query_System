# 🧠 LangChain SQL Query Assistant

This project is an AI-powered assistant that translates natural language questions into SQL queries, executes them on a MySQL database, and returns human-readable answers. It leverages LangChain, HuggingFace embeddings, Chroma vector store, and Mistral AI for language modeling.

# Pipeline
![sql_usecase](https://github.com/user-attachments/assets/3fa5ae04-8d6b-490e-be73-ce23ffae2c48)

# 📌 Features

Natural Language to SQL: Converts user questions into syntactically correct SQL queries.

Database Interaction: Connects to a MySQL database to execute generated queries.

Semantic Similarity: Utilizes HuggingFace embeddings and Chroma to select relevant examples for few-shot learning.

Dynamic Prompting: Constructs prompts dynamically using LangChain's FewShotPromptTemplate.

Structured Workflow: Employs LangGraph to orchestrate the workflow in defined steps.

# 🛠️ Installation 

- Install Dependencies: 
pip install -r requirements.txt
- Set Environment Variables:
Create a .env file in the project root:
  DB_PASSWORD=your_db_password
  HUGGING_FACE_KEY=your_huggingface_api_key
  MISTRAL_API_KEY=your_mistral_api_key

# 🧩 Project Structure

few_shots.py: Contains example question-query pairs for few-shot learning.

Smart_SQL_Engine.ipynb : Jupyter notebook of the code to run the system.

README.md: Project documentation.

.env: Environment variables (not included in the repository).

requirements.txt : File for the required installations 

database.sql : SQL code to generate the DB

app.py : Streamlit application





