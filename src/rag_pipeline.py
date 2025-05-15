from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import pandas as pd
from typing import List
from .vector_indexing import VectorIndex

class RAGPipeline:
    def __init__(self, df: pd.DataFrame, vector_index: VectorIndex, groq_api_key: str):
        """
        Initialize the RAG pipeline.
        
        Args:
            df (pd.DataFrame): Laptop DataFrame.
            vector_index (VectorIndex): FAISS index.
            groq_api_key (str): Groq API key.
        """
        try:
            self.df = df
            self.vector_index = vector_index
            self.llm = ChatGroq(model="llama3-8b-8192", groq_api_key=groq_api_key)
            
            # Prompt template
            self.prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template="""
                You are an e-commerce assistant for laptop recommendations.
                Use the provided laptop details to give a concise, accurate response.
                Focus on the user's needs (e.g., budget, use case) and recommend the best match.
                Include specific details (name, price, specs) and explain why it fits.
                Suggest alternatives if relevant.
                
                Laptop Details:
                {context}
                
                User Query: {question}
                
                Response:
                """
            )
            
            # LangChain pipeline
            self.chain = (
                {"context": self._retrieve_context, "question": RunnablePassthrough()}
                | self.prompt_template
                | self.llm
                | StrOutputParser()
            )
            
        except Exception as e:
            print(f"Error initializing RAG pipeline: {str(e)}")
            raise
    
    def _retrieve_context(self, query: str) -> str:
        """
        Retrieve laptop details for the query.
        
        Args:
            query (str): User query.
            
        Returns:
            str: Formatted context string.
        """
        try:
            distances, indices = self.vector_index.search(query, k=5)
            context = ""
            for idx in indices[0]:
                row = self.df.iloc[idx]
                context += (
                    f"- {row['name']} (₹{row['price']:,.0f}, {row['rating']} stars): "
                    f"{row['Processor']}, {row['RAM']}, {row['SSD']}, "
                    f"{row['Graphics']}, {row['Display_Size']} cm display, "
                    f"{row['Operating_System']}. URL: {row['url']}\n"
                )
            return context
            
        except Exception as e:
            print(f"Error retrieving context: {str(e)}")
            raise
    
    def run(self, query: str) -> str:
        """
        Run the RAG pipeline.
        
        Args:
            query (str): User query.
            
        Returns:
            str: Generated response.
        """
        try:
            response = self.chain.invoke(query)
            return response
        except Exception as e:
            print(f"Error running RAG pipeline: {str(e)}")
            raise