# 🌱 Lufa Farms Data Analyst AI Agent

An intelligent natural language to SQL query interface specifically designed for Lufa Farms' database analysis. This tool allows analysts to query the database using natural language, which is then converted into accurate SQL queries.

## 🚀 Features

- Natural language to SQL query conversion
- Interactive web interface built with Streamlit
- Support for both OpenAI GPT-4 and Anthropic Claude models
- Semantic search powered by OpenAI embeddings and FAISS
- Query history tracking
- Detailed schema context awareness
- Real-time query validation and refinement
- Interactive query results display

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **AI Models**: 
  - OpenAI GPT-4 Turbo
  - Anthropic Claude 3.5 Sonnet
  - OpenAI Ada Embeddings
- **Vector Search**: FAISS (Facebook AI Similarity Search)
- **Database**: MySQL
- **Key Libraries**:
  - `streamlit`
  - `openai`
  - `anthropic`
  - `faiss-cpu`
  - `pandas`
  - `numpy`
  - `sqlalchemy`
  - `python-dotenv`

## 📋 Prerequisites

- Python 3.8+
- MySQL Database access
- OpenAI API key
- Claude API key

## ⚙️ Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your credentials:
```
CLAUDE_API_KEY="your-claude-api-key"
OPENAI_API_KEY="your-openai-api-key"
DB_HOST="your-db-host"
DB_PORT="your-db-port"
DB_USER="your-db-user"
DB_PASSWORD="your-db-password"
DB_NAME="your-db-name"
```

5. Prepare your database metadata:
   - Place your `mysql_metadata.csv` file in the root directory
   - Place your `filtered_relationships.csv` file in the root directory

## 🚀 Running the Application

1. Activate your virtual environment if not already activated:
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Run the Streamlit app:
```bash
streamlit run chatbot.py
```

3. Open your browser and navigate to the provided local URL (typically http://localhost:8501)

## 💡 Usage

1. Select your preferred AI model in the sidebar (ChatGPT Turbo or Claude 3.5 Sonnet)
2. Enter your question in natural language in the text area
3. Click "Generate Query" to convert your question to SQL
4. Review and execute the generated SQL query
5. View the results in the interactive data table

## 🔒 Security

- API keys and database credentials are stored in the `.env` file
- The `.env` file is excluded from version control via `.gitignore`
- Database queries are validated before execution
- Input sanitization is implemented to prevent SQL injection

## 📝 Notes

- The application maintains a history of your recent queries
- You can toggle query details view in the settings
- The system automatically refines queries if initial generation fails
- Vector embeddings are cached for better performance

## 🤝 Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE.md file for details 