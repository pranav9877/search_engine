# Laptop Search Engine

A production-level search engine for e-commerce laptop data, built with FAISS, LangChain, RAG, and Streamlit.

## Features
- Natural language query support (e.g., "laptop under 35k for school work").
- Vector search using FAISS and `sentence-transformers`.
- RAG pipeline with Groq’s LLaMA model for contextual responses.
- Streamlit UI with price, brand, and use-case filters.
- Modular, scalable, and error-handled code.

## Setup
1. **Clone the repository**:
   ```bash
   git clone <repo_url>
   cd laptop_search_engine
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file with:
   ```
   GROQ_API_KEY=your_groq_api_key
   ```

4. **Prepare data**:
   Place `laptops.csv` in the `data/` directory.

5. **Run the application**:
   ```bash
   streamlit run src/app.py
   ```

## Usage
- Enter a query in the Streamlit UI (e.g., "best Windows laptop under 100k").
- Apply filters for price, brand, or use case.
- View results with detailed recommendations and laptop links.

## Deployment
- **Local**: Run via `streamlit run src/app.py`.
- **Hugging Face Spaces**: Deploy using Streamlit’s deployment guide.

## Requirements
- Python 3.8+
- 16 GB RAM recommended
- Optional: GPU for faster embedding generation
