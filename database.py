import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer



class PineconePipeline:
    def __init__(
        self,
        pinecone_api_key: str,
        google_api_key: str,
        environment: str,
        index_name: str,
        dimension: int = 384,
        embedding_model_name: str = "intfloat/e5-small-v2",
        device: str = "cpu"
    ):
        """
        Initialize the Pinecone pipeline with LangChain and Google AI integration.
        
        Args:
            pinecone_api_key: Pinecone API key
            google_api_key: Google AI API key
            environment: Pinecone environment
            index_name: Name of the Pinecone index
            dimension: Dimension of the embeddings
            embedding_model_name: Name of the embedding model
            device: Device to run the model on ('cuda' or 'cpu')
        """
        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        
        # Initialize LangChain with Gemini
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0,
            google_api_key=google_api_key
        )
        
        # Create Summarization Chain
        map_prompt_template = """
        Write a concise summary of the following text:
        "{text}"
        CONCISE SUMMARY:
        """
        
        combine_prompt_template = """
        Given the following extracted summaries and the question: "{question}"
        Please provide a comprehensive answer based on the summaries.

        Summaries:
        {text}

        COMPREHENSIVE ANSWER:
        """
        
        self.map_prompt = PromptTemplate(
            template=map_prompt_template,
            input_variables=["text"]
        )
        
        self.combine_prompt = PromptTemplate(
            template=combine_prompt_template,
            input_variables=["text", "question"]
        )
        
        self.chain = load_summarize_chain(
            llm=self.llm,
            chain_type="map_reduce",
            map_prompt=self.map_prompt,
            combine_prompt=self.combine_prompt,
            verbose=True
        )
        
        # Create index if it doesn't exist
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=environment
                )
            )
        
        self.index = self.pc.Index(index_name)
        self.embedding_model = SentenceTransformer(
            model_name_or_path=embedding_model_name,
            device=device
        )
        
    def prepare_vectors(
        self,
        chunks_df: pd.DataFrame,
        batch_size: int = 100
    ) -> List[Dict]:
        """
        Prepare vectors for Pinecone insertion from a DataFrame.
        
        Args:
            chunks_df: DataFrame containing text chunks and metadata
            batch_size: Size of batches for processing
        
        Returns:
            List of dictionaries formatted for Pinecone upsert
        """
        vectors = []
        
        for i in tqdm(range(0, len(chunks_df), batch_size), desc="Preparing vectors"):
            batch = chunks_df.iloc[i:i + batch_size]
            
            # Generate embeddings for the batch
            texts = batch['sentence_chunk'].tolist()
            embeds = self.embedding_model.encode(texts)
            
            # Create vector records
            for j, embed in enumerate(embeds):
                vectors.append({
                    'id': f"chunk_{i+j}",
                    'values': embed.tolist(),
                    'metadata': {
                        'text': batch.iloc[j]['sentence_chunk'],
                        'page_number': int(batch.iloc[j]['page_number']),
                        'chunk_token_count': float(batch.iloc[j]['chunk_token_count'])
                    }
                })
        
        return vectors
    
    def upsert_vectors(
        self,
        vectors: List[Dict],
        batch_size: int = 100
    ) -> None:
        """
        Upsert vectors to Pinecone in batches.
        
        Args:
            vectors: List of vector dictionaries
            batch_size: Size of batches for upserting
        """
        for i in tqdm(range(0, len(vectors), batch_size), desc="Upserting to Pinecone"):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
            time.sleep(0.5)  # Rate limiting
    
    def query(
        self,
        query_text: str,
        top_k: int = 5,
        filter: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Query the Pinecone index.
        
        Args:
            query_text: Text to search for
            top_k: Number of results to return
            filter: Optional metadata filters
        
        Returns:
            List of matching results with scores and metadata
        """
        # Generate embedding for query
        query_embedding = self.embedding_model.encode(query_text).tolist()
        
        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter
        )
        
        return results['matches']
    
    def query_and_get_results(
        self,
        query_text: str,
        top_k: int = 5,
        filter: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Query the Pinecone index and return the results.
        
        Args:
            query_text: Text to search for
            top_k: Number of results to return
            filter: Optional metadata filters
        
        Returns:
            List of matching results with scores and metadata
        """
        # Generate embedding for query
        query_embedding = self.embedding_model.encode(query_text).tolist()
        
        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter
        )
        
        return results['matches']
    
    
    def delete_all(self) -> None:
        """Delete all vectors from the index."""
        self.index.delete(delete_all=True)

    def query_and_summarize(
        self,
        query_text: str,
        top_k: int = 5,
        filter: Optional[Dict] = None,
    ) -> Dict:
        """
        Query Pinecone and generate a summary using LangChain.
        
        Args:
            query_text: Text to search for
            top_k: Number of results to return
            filter: Optional metadata filters
            
        Returns:
            Dictionary containing both raw results and AI-generated summary
        """
        # Get vector search results
        results = self.query(query_text, top_k, filter)
        
        # Convert results to LangChain documents
        documents = [
            Document(
                page_content=match['metadata']['text'],
                metadata={
                    'score': match['score'],
                    'page_number': match['metadata']['page_number']
                }
            )
            for match in results
        ]
        
        # Generate summary using LangChain
        summary = self.chain.run(
            input_documents=documents,
            question=query_text
        )
        
        return {
            'raw_results': results,
            'summary': summary,
            'source_pages': [doc.metadata['page_number'] for doc in documents]
        }

    def batch_summarize(
        self,
        queries: List[str],
        top_k: int = 5,
        filter: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Process multiple queries and generate summaries for each.
        
        Args:
            queries: List of query texts
            top_k: Number of results per query
            filter: Optional metadata filters
            
        Returns:
            List of dictionaries containing results and summaries
        """
        results = []
        for query in tqdm(queries, desc="Processing queries"):
            summary_result = self.query_and_summarize(
                query_text=query,
                top_k=top_k,
                filter=filter
            )
            results.append(summary_result)
        return results

# Example usage
def main():
    # Initialize pipeline
    pipeline = PineconePipeline(
        pinecone_api_key="pcsk_43sajZ_MjcXR2yN5cAcVi8RARyB6i3NP3wLTnTLugbUcN9cUU4q5EfNmuwLPkmxAvykk9o",
        google_api_key="lRuqeXqQ_AljvIAj_qNBJdA8c9U_AmnNmH31F9WyuhY ",  # Provide your Google API key

        environment="us-east-1",  # Use the appropriate AWS region
        index_name="pdf-embeddings"
    )
    
    
    # Example query
    results = pipeline.query(
        query_text="Under section 42, the dissolution of the firm depends",
        top_k=5
    )
    
    # Print results
    for match in results:
        print(f"Score: {match['score']:.4f}")
        print("Text:", match['metadata']['text'])
        print("Page:", match['metadata']['page_number'])
        print()

if __name__ == "__main__":
    main()