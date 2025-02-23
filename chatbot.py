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
    
    # Delete existing index files if they exist
    if os.path.exists(FAISS_INDEX_PATH):
        os.remove(FAISS_INDEX_PATH)
    if os.path.exists(EMBEDDINGS_PATH):
        os.remove(EMBEDDINGS_PATH)
    
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
            description += f" Purpose: {comment}"
        
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
            description += f" Purpose: {comment}"
        
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

def generate_sql_query(nl_query, context, composite_schema, relationships, model_choice, max_tokens=2000):
    """
    Generate a SQL query using the LLM.
    Uses a condensed schema to keep the prompt size within limits.
    """
    condensed_schema = get_condensed_schema(composite_schema)
    
    # Get relevant business rules
    relevant_rules = get_relevant_rules(nl_query, context)
    business_rules = "\n".join([f"{i+1}. {rule}" for i, rule in enumerate(relevant_rules)])
    
    # Check if this is a lateness query
    is_lateness_query = any(word in nl_query.lower() for word in ['late', 'lateness', 'delay'])
    
    if is_lateness_query:
        prompt = f"""You MUST generate a SQL query for a delivery lateness question using EXACTLY the template provided below.
        
STRICT TEMPLATE - You can ONLY modify:
1. The date range in the INNER JOIN condition (BETWEEN "start_date" AND "end_date")
2. The minute threshold in the WHERE clause TIME_TO_SEC calculation (the X in <= -(X*60))

Template to follow EXACTLY (copy and paste this exactly, only replacing the date range and X value):
SELECT 
            ww_orders.delivery_date AS 'DELIVERY DATE',
            IF(ww_orders.order_number>0, 
            'HD', 
            CASE
                WHEN drop_instance.type=1 THEN 'PUP'
                WHEN drop_instance.type=2 THEN 'PICKUP'
                WHEN drop_instance.type=3 THEN 'EMBASSY'
                WHEN drop_instance.type IS NULL THEN 'BREAK'
            END
            ) AS 'DELIVERY TYPE',
            
            CASE
            WHEN ww_steps.status=1 THEN 'DONE'
            WHEN ww_steps.status=2 THEN 'CANCELED'
            WHEN ww_steps.status=0 THEN 'INCOMPLETED'
            END AS 'DELIVERY STATUS',
            IF(ww_orders.order_number>0, user_home_deliveries.city_name, droppoints.city_name) AS 'DELIVERY ADDRESS',
            
            ww_steps.limit_datetime AS 'TIME LIMIT FOR DELIVERY',
            ww_steps.completed_datetime AS 'REAL DELIVERY TIME',
            TIMEDIFF(ww_steps.limit_datetime,ww_steps.completed_datetime) AS 'DELIVERY LATENESS',
            CASE
                WHEN ww_steps.completed_datetime > ww_steps.limit_datetime AND (ww_orders.order_number > 0) THEN 'YES'
                WHEN ww_steps.completed_datetime > ww_steps.limit_datetime AND (ww_orders.order_number IS NULL AND drop_instance.type IS NOT NULL) THEN 'YES'
                ELSE 'NO'
            END AS 'LATE',
            TIME_TO_SEC(TIMEDIFF(ww_steps.limit_datetime,ww_steps.completed_datetime))
        FROM 
            ww_steps
        INNER JOIN ww_orders ON (ww_orders.order_id = ww_steps.order_id AND ww_orders.delivery_date BETWEEN "start_date" AND "end_date") 
        LEFT JOIN user_home_deliveries ON (user_home_deliveries.order_id = ww_orders.order_number)
        LEFT JOIN drop_instance ON (drop_instance.drop_instance_id = ww_orders.drop_instance_id)
        LEFT JOIN droppoints ON (droppoints.droppoint_id = drop_instance.droppoint_id)
        WHERE
            ww_steps.type IN (0,2) AND 
            TIME_TO_SEC(TIMEDIFF(ww_steps.limit_datetime,ww_steps.completed_datetime)) <= -(X*60)

        GROUP BY 
            ww_steps.step_id
        ORDER BY 
            ww_orders.delivery_date,
            ww_steps.order ASC

IMPORTANT RULES:
1. Use full table names (e.g., ww_steps, ww_orders) - DO NOT use aliases
2. Use double quotes for date values ("YYYY-MM-DD")
3. Keep all spacing and formatting exactly as shown
4. The date range goes in the INNER JOIN condition with ww_orders
5. The minute threshold goes in the WHERE clause TIME_TO_SEC calculation

Natural language query: {nl_query}

Generate the SQL query by using the template above and ONLY modifying:
1. Replace "start_date" and "end_date" with the actual dates from the question (e.g., "2025-01-05" AND "2025-01-06")
2. Replace X with the minute threshold from the question (e.g., 20 for 20 minutes)
Do not change anything else - copy the template exactly and only replace those values."""
    else:
        prompt = f"""Convert the following natural language query into a valid MySQL SQL query using only the provided schema.

Available Schema (condensed):
{json.dumps(condensed_schema, indent=2)}

Relationships:
{json.dumps(relationships, indent=2)}

Context:
{context}

IMPORTANT BUSINESS RULES:
{business_rules}

Query: {nl_query}

IMPORTANT TECHNICAL RULES:
1. Always qualify column names with their table names (e.g., table_name.column_name)
2. Only use columns that exist in the provided schema
3. Do not invent new column names
4. Ensure proper joins based on the relationships
5. For subqueries, ensure column references are valid

Return ONLY the SQL query with no markdown formatting or commentary.
"""
    try:
        if model_choice == "ChatGPT Turbo":
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are an expert SQL generator. For lateness queries, you MUST use the exact template provided with only date and threshold modifications allowed."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens
            )
            sql_query = response.choices[0].message.content.strip()
        else:
            anthro_client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
            response = anthro_client.messages.create(
                model="claude-3-opus-20240229",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            sql_query = response.content[0].text.strip()
        return extract_sql(sql_query)
    except Exception as e:
        st.error(f"Error generating SQL: {str(e)}")
        return None

def refine_sql_query(previous_query, error_message, context, composite_schema, relationships, model_choice, max_tokens=2000):
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
        if model_choice == "ChatGPT Turbo":
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert SQL generator. Correct the SQL query based on the feedback provided."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens
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
        engine = create_engine(f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")
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
    
    # Sidebar: model selection and explanation mode
    st.sidebar.title("Settings")
    st.session_state.explanation_mode = st.sidebar.checkbox("Show Query Details", value=False)
    model_choice = st.sidebar.selectbox("Select Model", ["ChatGPT Turbo", "Claude 3.5 Sonnet"])
    
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
    
    if st.button("Generate Query"):
        if nl_query.strip() and nl_query not in st.session_state.query_history:
            st.session_state.query_history.insert(0, nl_query)
            st.session_state.query_history = st.session_state.query_history[:10]
        
        with st.spinner("Generating SQL query..."):
            context = get_relevant_context(nl_query, st.session_state.embeddings, st.session_state.index, st.session_state.df_schema, top_k=st.session_state.context_top_k)
            sql_query = generate_sql_query(
                nl_query,
                context,
                st.session_state.composite_schema,
                st.session_state.df_relationships,
                model_choice
            )
            valid, message = validate_columns(sql_query, st.session_state.table_columns)
            retry_count = 0
            max_retries = 3
            
            while not valid and retry_count < max_retries:
                st.warning(f"Validation failed: {message}. Refining the query (Attempt {retry_count+1} of {max_retries})")
                refined_query = refine_sql_query(
                    sql_query,
                    message,
                    context,
                    st.session_state.composite_schema,
                    st.session_state.df_relationships,
                    model_choice
                )
                if not refined_query:
                    break
                sql_query = refined_query
                valid, message = validate_columns(sql_query, st.session_state.table_columns)
                retry_count += 1
            
            if not valid:
                st.error(f"SQL query could not be refined after {max_retries} attempts: {message}.")
                # Optionally, re-run similarity search with increased context (top_k)
                st.warning("Refinement attempts failed. Re-running similarity search with more context...")
                st.session_state.context_top_k = 10
                context = get_relevant_context(nl_query, st.session_state.embeddings, st.session_state.index, st.session_state.df_schema, top_k=st.session_state.context_top_k)
                sql_query = generate_sql_query(
                    nl_query,
                    context,
                    st.session_state.composite_schema,
                    st.session_state.df_relationships,
                    model_choice
                )
                valid, message = validate_columns(sql_query, st.session_state.table_columns)
                if not valid:
                    st.error(f"SQL query could not be refined: {message}")
                else:
                    st.session_state.sql_query = sql_query
                    st.session_state.generated_context = context
                    st.session_state.sql_generated = True
            else:
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
