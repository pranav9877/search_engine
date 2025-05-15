import streamlit as st
import pandas as pd
from src.data_processing import load_and_preprocess_data
from src.vector_indexing import VectorIndex
from src.rag_pipeline import RAGPipeline
from dotenv import load_dotenv
import os

def main():
    st.title("Laptop Search Engine")
    st.write("Find the perfect laptop based on your needs!")

    # Load environment variables
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("Groq API key not found in .env file.")
        return

    # Initialize components
    @st.cache_resource
    def initialize_pipeline():
        try:
            # Load and preprocess data
            df, descriptions = load_and_preprocess_data("data/laptops.csv")
            
            # Build vector index
            vector_index = VectorIndex()
            vector_index.build_index(descriptions)
            
            # Initialize RAG pipeline
            rag_pipeline = RAGPipeline(df, vector_index, groq_api_key)
            return df, rag_pipeline
        except Exception as e:
            st.error(f"Error initializing pipeline: {str(e)}")
            return None, None

    df, rag_pipeline = initialize_pipeline()
    if df is None or rag_pipeline is None:
        return

    # User input
    query = st.text_input("Enter your query (e.g., 'laptop under 35k for school work')")
    
    # Filters
    st.sidebar.header("Filters")
    max_price = st.sidebar.slider("Max Price (₹)", 0, 200000, 100000, step=5000)
    brands = sorted(df['name'].str.split().str[0].unique())
    selected_brands = st.sidebar.multiselect("Brands", brands, default=brands)
    use_case = st.sidebar.selectbox("Use Case", ["Any", "School Work", "Gaming", "Professional", "AI/ML"])

    if st.button("Search"):
        if not query:
            st.warning("Please enter a query.")
            return
        
        try:
            # Apply filters
            filtered_df = df[df['price'] <= max_price]
            if selected_brands:
                filtered_df = filtered_df[filtered_df['name'].str.split().str[0].isin(selected_brands)]
            
            # Modify query based on use case
            if use_case != "Any":
                query = f"{query} for {use_case.lower()}"
            
            # Run RAG pipeline
            response = rag_pipeline.run(query)
            st.markdown("### Results")
            st.write(response)
            
            # Display filtered laptops
            st.markdown("### Matching Laptops")
            for idx in filtered_df.index[:5]:  # Show top 5
                row = filtered_df.loc[idx]
                st.write(
                    f"- **{row['name']}** (₹{row['price']:,.0f}, {row['rating']} stars): "
                    f"{row['Processor']}, {row['RAM']}, {row['SSD']}, {row['Graphics']}, "
                    f"{row['Display_Size']} cm. [Link]({row['url']})"
                )
                
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main()