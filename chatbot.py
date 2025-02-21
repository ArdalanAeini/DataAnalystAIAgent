import os
import re
import json
import faiss
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import openai
import anthropic
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
from openai import OpenAI

#############################################
# Utility Functions & Initialization
#############################################

def initialize_app():
    """Load environment variables and initialize session state."""
    load_dotenv()
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.sql_query = ""
        st.session_state.sql_generated = False
        st.session_state.explanation_mode = False
        st.session_state.generated_context = ""
        st.session_state.query_history = []
        st.session_state.selected_query = None
        st.session_state.nl_query = ""
        st.session_state.retry_count = 0  # For tracking refinement iterations
        st.session_state.context_top_k = 5  # initial top_k for similarity search
    return {
        "host": st.secrets["DB_HOST"] or os.getenv("DB_HOST"),
        "port": st.secrets["DB_PORT"] or os.getenv("DB_PORT"),
        "user": st.secrets["DB_USER"] or os.getenv("DB_USER"),
        "password": st.secrets["DB_PASSWORD"] or os.getenv("DB_PASSWORD"),
        "database": st.secrets["DB_NAME"] or os.getenv("DB_NAME")
    }

#############################################
# Schema Loading & Composite Schema Construction
#############################################

def load_database_schema():
    """
    Load CSV files:
      - mysql_metadata.csv: detailed column info
      - filtered_relationships.csv: relationships
    Returns dataframes, basic mappings, and a composite schema.
    """
    try:
        df_schema = pd.read_csv("mysql_metadata.csv")
        df_relationships = pd.read_csv("filtered_relationships.csv")
        valid_tables = df_schema["Table Name"].unique().tolist()
        table_columns = df_schema.groupby("Table Name")["Column Name"].apply(list).to_dict()
        composite_schema = build_composite_schema(df_schema)
        return df_schema, None, df_relationships, valid_tables, table_columns, composite_schema
    except Exception as e:
        st.error(f"Error loading schema: {str(e)}")
        return None, None, None, None, None, None

def build_composite_schema(df_schema):
    """
    For each table, build a composite dictionary including:
      - A list of column details from df_schema.
    Returns a dictionary mapping table names to their composite info.
    """
    composite_schema = {}
    for table in df_schema["Table Name"].unique():
        cols_df = df_schema[df_schema["Table Name"] == table]
        columns_info = []
        for _, row in cols_df.iterrows():
            col_info = {
                "Column Name": row["Column Name"],
                "Data Type": row["Data Type"],
                "Null Allowed": row["Null Allowed"],
                "Key": row["Key"],
                "Default Value": row["Default Value"],
                "Extra": row["Extra"],
                "Comment": row["Comment"]
            }
            columns_info.append(col_info)
        composite_schema[table] = {
            "columns": columns_info
        }
    return composite_schema

def get_condensed_schema(composite_schema):
    """
    Create a condensed version of the composite schema that includes for each table:
      - List of column names.
    """
    condensed = {}
    for table, info in composite_schema.items():
        condensed[table] = {
            "columns": [col_info["Column Name"] for col_info in info["columns"]]
        }
    return condensed

#############################################
# FAISS Index & Context Retrieval
#############################################

def get_embedding(text, model="text-embedding-ada-002"):
    """Get embedding from OpenAI API."""
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"] or os.getenv("OPENAI_API_KEY"))
    text = text.replace("\n", " ")
    response = client.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding

def setup_faiss_index(df_schema):
    """Initialize or load FAISS index using OpenAI embeddings."""
    FAISS_INDEX_PATH = "faiss_index.bin"
    EMBEDDINGS_PATH = "embeddings.pkl"
    
    # Try to load existing index and embeddings
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(EMBEDDINGS_PATH):
        try:
            index = faiss.read_index(FAISS_INDEX_PATH)
            with open(EMBEDDINGS_PATH, "rb") as f:
                embeddings = pickle.load(f)
            return embeddings, index
        except Exception as e:
            st.warning(f"Failed to load existing index files, creating new ones: {str(e)}")
    
    # If we get here, either files don't exist or loading failed
    texts = []
    for _, row in df_schema.iterrows():
        # Create a natural language description of the column
        key_info = row['Key'] if pd.notna(row['Key']) else None
        default_value = row['Default Value'] if pd.notna(row['Default Value']) else None
        extra_info = row['Extra'] if pd.notna(row['Extra']) else None
        comment = row['Comment'] if pd.notna(row['Comment']) else None
        
        description = f"""In the {row['Table Name']} table, there is a column named {row['Column Name']} which stores {row['Data Type']} data. 
        This column {' cannot be null' if row['Null Allowed'] == 'NO' else 'allows null values'}."""
        
        if key_info:
            if key_info == 'PRI':
                description += f" It serves as the primary key for the table."
            elif key_info == 'MUL':
                description += f" It is part of an index or foreign key relationship."
            elif key_info == 'UNI':
                description += f" It has a unique constraint, meaning each value must be unique."
            else:
                description += f" It has a {key_info} key type."
        
        if default_value is not None:
            description += f" If no value is specified, it defaults to {default_value}."
        
        if extra_info:
            description += f" Additional properties: {extra_info}."
        
        if comment:
            description += f" Purpose of this column: {comment}"
        
        texts.append(description)
    
    # Get embeddings for all texts
    embeddings = []
    for text in texts:
        embedding = get_embedding(text)
        embeddings.append(embedding)
    
    # Convert to numpy array
    embeddings = np.array(embeddings, dtype=np.float32)
    
    # Create and populate FAISS index
    d = len(embeddings[0])  # dimensionality of embeddings
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    
    # Save the new index and embeddings
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(embeddings, f)
    
    return embeddings, index

def get_relevant_context(query, embeddings, index, df_schema, top_k=5):
    """Retrieve relevant context from FAISS based on the query using OpenAI embeddings."""
    query_embedding = get_embedding(query)
    query_embedding_array = np.array([query_embedding], dtype=np.float32)
    _, idxs = index.search(query_embedding_array, top_k)
    
    # Build the same full context that was vectorized
    contexts = []
    for idx in idxs[0]:
        row = df_schema.iloc[idx]
        key_info = row['Key'] if pd.notna(row['Key']) else None
        default_value = row['Default Value'] if pd.notna(row['Default Value']) else None
        extra_info = row['Extra'] if pd.notna(row['Extra']) else None
        comment = row['Comment'] if pd.notna(row['Comment']) else None
        
        description = f"""In the {row['Table Name']} table, there is a column named {row['Column Name']} which stores {row['Data Type']} data. 
        This column {' cannot be null' if row['Null Allowed'] == 'NO' else 'allows null values'}."""
        
        if key_info:
            if key_info == 'PRI':
                description += f" It serves as the primary key for the table."
            elif key_info == 'MUL':
                description += f" It is part of an index or foreign key relationship."
            elif key_info == 'UNI':
                description += f" It has a unique constraint, meaning each value must be unique."
            else:
                description += f" It has a {key_info} key type."
        
        if default_value is not None:
            description += f" If no value is specified, it defaults to {default_value}."
        
        if extra_info:
            description += f" Additional properties: {extra_info}."
        
        if comment:
            description += f" Purpose of this column: {comment}"
        
        contexts.append(description)
    
    return "\n\n".join(str(context) for context in contexts if context is not None)

#############################################
# SQL Generation, Validation, and Feedback
#############################################

def extract_sql(query_text):
    """
    Remove markdown code blocks, backticks, and any leading "sql" text.
    Returns a clean SQL query string.
    """
    cleaned = re.sub(r'```(?:sql)?', '', query_text, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r'```', '', cleaned).strip()
    if cleaned.lower().startswith("sql"):
        cleaned = cleaned[3:].strip()
    return cleaned

def load_instruction_rules():
    """Load business rules and instructions from CSV."""
    try:
        df_rules = pd.read_csv("instruction_rules.csv")
        # Group rules by context
        rules_by_context = df_rules.groupby("Context").apply(
            lambda x: x[["Rule", "Description"]].to_dict("records")
        ).to_dict()
        return rules_by_context
    except Exception as e:
        st.error(f"Error loading instruction rules: {str(e)}")
        return {}

def get_relevant_rules(nl_query, context):
    """
    Determine which rules are relevant based on the query and context.
    Returns a list of relevant rule descriptions.
    """
    rules = []
    # Load rules if not in session state
    if 'instruction_rules' not in st.session_state:
        st.session_state.instruction_rules = load_instruction_rules()
    
    # Check each context for relevance
    for context_type, context_rules in st.session_state.instruction_rules.items():
        # If the context type or related terms are in the query or context
        if (context_type.lower() in nl_query.lower() or 
            context_type.lower() in context.lower()):
            # Add all rules for this context
            rules.extend([rule["Description"] for rule in context_rules])
    
    return rules

def analyze_query_clarity(nl_query, context, composite_schema, relationships, model_choice, max_tokens=1000):
    """
    Analyze the natural language query for ambiguity and request clarification if needed.
    Returns (needs_clarification, clarification_question)
    """
    condensed_schema = get_condensed_schema(composite_schema)
    
    prompt = f"""Analyze this natural language query and determine if any clarification is needed.
If the query is clear and has all needed information, respond with "CLEAR: The query is clear."
If clarification is needed, respond with "NEEDS_CLARIFICATION:" followed by a specific question to ask the user.

Natural Language Query: {nl_query}

Available Schema:
{json.dumps(condensed_schema, indent=2)}

Context:
{context}

Consider the following aspects:
1. Time periods (if mentioned, are they specific enough?)
2. Metrics or calculations (are they well defined?)
3. Filtering conditions (are they specific enough?)
4. Sorting/ordering (is the order clear?)
5. Grouping (is it clear how to group the data?)
6. Ambiguous terms (are all terms clearly mapped to database fields?)

Provide ONLY one of these two response formats:
CLEAR: The query is clear.
or
NEEDS_CLARIFICATION: <your specific question here>"""

    try:
        if model_choice == "O1 Reasoning Model":
            client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"] or os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="o1",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing natural language queries for ambiguity and determining if clarification is needed."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=max_tokens
            )
            result = response.choices[0].message.content.strip()
        else:
            anthro_client = anthropic.Anthropic(api_key=st.secrets["CLAUDE_API_KEY"] or os.getenv("CLAUDE_API_KEY"))
            response = anthro_client.messages.create(
                model="claude-3-opus-20240229",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            result = response.content[0].text.strip()

        if result.startswith("CLEAR:"):
            return False, None
        elif result.startswith("NEEDS_CLARIFICATION:"):
            return True, result[len("NEEDS_CLARIFICATION:"):].strip()
        return False, None
    except Exception as e:
        st.error(f"Error analyzing query clarity: {str(e)}")
        return False, None

def generate_sql_query(nl_query, context, composite_schema, relationships, model_choice="Claude 3.5 Sonnet", max_tokens=2000, clarification_response=None):
    """
    Generate a SQL query using Claude.
    Uses a condensed schema to keep the prompt size within limits.
    """
    condensed_schema = get_condensed_schema(composite_schema)
    
    # Get relevant business rules
    relevant_rules = get_relevant_rules(nl_query, context)
    business_rules = "\n".join([f"{i+1}. {rule}" for i, rule in enumerate(relevant_rules)])
    
    # Include clarification in the prompt if provided
    clarification_text = f"\nClarification provided: {clarification_response}" if clarification_response else ""
    
    prompt = f"""Convert the following natural language query into a MySQL SQL query using only the provided schema.

Available Schema (condensed):
{json.dumps(condensed_schema, indent=2)}

Relationships:
{json.dumps(relationships, indent=2)}

Context:
{context}{clarification_text}

IMPORTANT BUSINESS RULES:
{business_rules}

Query: {nl_query}

IMPORTANT TECHNICAL RULES:
1. Only use columns that exist in the provided schema
2. ALWAYS use FULL table names (e.g., 'products.name', NOT 'p.name') - Table aliases are strictly forbidden
3. Do not invent new column names
4. Ensure proper joins based on the relationships
5. For subqueries, ensure column references are valid
6. Example of correct table references:
   - Use: products.name, products.price, product_stats.qty_sold
   - DO NOT use: p.name, ps.qty_sold, etc.
7. SAFETY MEASURES:
   - DO NOT generate queries that modify tables (CREATE, ALTER, DROP, TRUNCATE)
   - DO NOT generate queries that modify data (INSERT, UPDATE, DELETE)
   - ONLY generate SELECT queries for data retrieval
   - Any attempt to modify database structure or data is strictly forbidden

Return ONLY the SQL query with no markdown formatting or commentary.
"""
    try:
        if model_choice == "O1 Reasoning Model":
            client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"] or os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="o1",
                messages=[
                    {"role": "system", "content": "You are an expert SQL generator with strong reasoning capabilities. Generate precise and efficient MySQL queries based on the schema and requirements provided."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=max_tokens
            )
            sql_query = response.choices[0].message.content.strip()
        else:
            anthro_client = anthropic.Anthropic(api_key=st.secrets["CLAUDE_API_KEY"] or os.getenv("CLAUDE_API_KEY"))
            response = anthro_client.messages.create(
                model="claude-3-opus-20240229",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            sql_query = response.content[0].text.strip()
        cleaned_query = extract_sql(sql_query)
        return cleaned_query
    except Exception as e:
        st.error(f"Error generating SQL: {str(e)}")
        return None

def refine_sql_query(previous_query, error_message, context, composite_schema, relationships, model_choice, max_tokens=400):
    """
    Refine the SQL query based on error feedback.
    """
    condensed_schema = get_condensed_schema(composite_schema)
    prompt = f"""The previously generated SQL query:
{previous_query}

returned the following error: {error_message}

Using the provided schema details below:
Available Schema (condensed):
{json.dumps(condensed_schema, indent=2)}

Relationships:
{json.dumps(relationships, indent=2)}

Context:
{context}

Please refine and correct the SQL query. Return ONLY the corrected SQL query with no additional commentary.
"""
    try:
        if model_choice == "O1 Reasoning Model":
            client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"] or os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="o1",
                messages=[
                    {"role": "system", "content": "You are an expert SQL generator. Correct the SQL query based on the feedback provided."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=max_tokens
            )
            refined = response.choices[0].message.content.strip()
        else:
            anthro_client = anthropic.Client(api_key=os.getenv("CLAUDE_API_KEY"))
            response = anthro_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            refined = response.completion.strip()
        return extract_sql(refined)
    except Exception as e:
        st.error(f"Error refining SQL: {str(e)}")
        return None

def validate_columns(sql_query, table_columns):
    """
    Ensure that all table.column references in the SQL query exist in the schema.
    """
    pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)'
    refs = re.findall(pattern, sql_query)
    for table, col in refs:
        if table not in table_columns or col not in table_columns[table]:
            return False, f"Invalid column reference: {table}.{col}"
    return True, "Valid query"

#############################################
# SQL Query Execution
#############################################

def execute_query(sql_query, db_config):
    """
    Execute the SQL query using SQLAlchemy.
    """
    try:
        engine = create_engine(
            f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
        with engine.connect() as conn:
            result = conn.execute(text(sql_query))
            df_results = pd.DataFrame(result.fetchall(), columns=result.keys())
        return df_results, None
    except Exception as e:
        return None, str(e)

#############################################
# Main Application Interface
#############################################

def main():
    st.title("🌱 Lufa Farms Data Analyst AI Agent")
    db_config = initialize_app()
    
    # Initialize session state for clarification
    if 'awaiting_clarification' not in st.session_state:
        st.session_state.awaiting_clarification = False
        st.session_state.clarification_question = None
        st.session_state.original_query = None
        st.session_state.clarification_response = None
    
    # Sidebar: model selection and explanation mode
    st.sidebar.title("Settings")
    st.session_state.explanation_mode = st.sidebar.checkbox("Show Query Details", value=False)
    model_choice = st.sidebar.selectbox("Select Model", ["O1 Reasoning Model", "Claude 3.5 Sonnet"])
    
    # Load schema if not already initialized
    if not st.session_state.initialized:
        with st.spinner("Loading database schema..."):
            df_schema, _, df_relationships, valid_tables, table_columns, composite_schema = load_database_schema()
            if df_schema is None:
                st.error("Failed to load database schema.")
                return
            embeddings, index = setup_faiss_index(df_schema)
            st.session_state.df_schema = df_schema
            st.session_state.df_relationships = df_relationships.to_dict('records')
            st.session_state.valid_tables = valid_tables
            st.session_state.table_columns = table_columns
            st.session_state.composite_schema = composite_schema
            st.session_state.embeddings = embeddings
            st.session_state.index = index
            st.session_state.instruction_rules = load_instruction_rules()
            st.session_state.initialized = True
    
    # Query history dropdown
    if st.session_state.query_history:
        selected_history = st.selectbox("Previous queries:", [""] + st.session_state.query_history, key="history_select")
        if selected_history and selected_history != st.session_state.get("nl_query", ""):
            st.session_state.nl_query = selected_history
    
    # Main natural language query input
    nl_query = st.text_area(
        "Enter your question:", 
        value=st.session_state.get("nl_query", ""),
        placeholder="e.g., Show me the top 10 products ordered in the last month",
        key="nl_query"
    )
    
    # Add clarification input if needed
    if st.session_state.awaiting_clarification:
        st.info(st.session_state.clarification_question)
        clarification_response = st.text_input("Please provide clarification:", key="clarification_input")
        if st.button("Submit Clarification"):
            st.session_state.clarification_response = clarification_response
            st.session_state.awaiting_clarification = False
            # Rerun with clarification
            with st.spinner("Generating SQL query..."):
                context = get_relevant_context(st.session_state.original_query, st.session_state.embeddings, st.session_state.index, st.session_state.df_schema)
                sql_query = generate_sql_query(
                    st.session_state.original_query,
                    context,
                    st.session_state.composite_schema,
                    st.session_state.df_relationships,
                    model_choice,
                    clarification_response=clarification_response
                )
                if sql_query:
                    st.session_state.sql_query = sql_query
                    st.session_state.generated_context = context
                    st.session_state.sql_generated = True
    
    elif st.button("Generate Query"):
        if nl_query.strip() and nl_query not in st.session_state.query_history:
            st.session_state.query_history.insert(0, nl_query)
            st.session_state.query_history = st.session_state.query_history[:10]
        
        with st.spinner("Analyzing query..."):
            context = get_relevant_context(nl_query, st.session_state.embeddings, st.session_state.index, st.session_state.df_schema)
            needs_clarification, clarification_question = analyze_query_clarity(
                nl_query,
                context,
                st.session_state.composite_schema,
                st.session_state.df_relationships,
                model_choice
            )
            
            if needs_clarification:
                st.session_state.awaiting_clarification = True
                st.session_state.clarification_question = clarification_question
                st.session_state.original_query = nl_query
                st.rerun()
            
            else:
                with st.spinner("Generating SQL query..."):
                    sql_query = generate_sql_query(
                        nl_query,
                        context,
                        st.session_state.composite_schema,
                        st.session_state.df_relationships,
                        model_choice
                    )
                    if sql_query:
                        st.session_state.sql_query = sql_query
                        st.session_state.generated_context = context
                        st.session_state.sql_generated = True
    
    if st.session_state.sql_generated:
        if st.session_state.explanation_mode:
            st.subheader("Query Context")
            st.text_area("Relevant Schema Context:", st.session_state.generated_context, height=100, key="context_area")
        st.subheader("Generated SQL Query")
        sql_query = st.text_area("SQL Query (editable):", st.session_state.sql_query, height=150, key="sql_area")
        
        if st.button("Execute Query"):
            with st.spinner("Executing query..."):
                results, error = execute_query(sql_query, db_config)
                if error:
                    st.error(f"Query execution failed: {error}")
                elif results is not None:
                    st.subheader("Query Results")
                    st.dataframe(results)

if __name__ == "__main__":
    main()
