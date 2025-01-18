
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


class PineconePipelineBetter:
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

        map_prompt_template = """
        Analyze the following legal text segment and extract key information:
        
        TEXT: "{text}"
        
        Instructions:
        1. Maintain all legal terminology exactly as written
        2. Preserve section numbers and references
        3. Keep all specific conditions and requirements
        4. Include any mentioned time periods or deadlines
        
        DETAILED ANALYSIS:
        """
        
        combine_prompt_template = """
        Based on the following excerpts from legal documents and the question: "{question}"
        
        EXCERPTS:
        {text}
        
        Instructions:
        1. Synthesize a comprehensive answer that connects relevant sections
        2. Maintain precise legal language from the source material
        3. Reference specific sections and subsections where applicable
        4. If there are seemingly disconnected pieces of information, explain their relationship
        5. Highlight any conditions or exceptions that span multiple excerpts
        
        COMPREHENSIVE LEGAL ANALYSIS:
        """
        
        self.map_prompt = PromptTemplate(
            template=map_prompt_template,
            input_variables=["text"]
        )
        
        self.combine_prompt = PromptTemplate(
            template=combine_prompt_template,
            input_variables=["text", "question"]
        )
    
    # Using stuff chain for small chunks to maintain better context
        self.chain = load_summarize_chain(
            llm=self.llm,
            chain_type="stuff",  # Changed from map_reduce to stuff for better handling of small chunks
            prompt=self.combine_prompt,
            verbose=True
        )

        self.index = self.pc.Index(index_name)
        self.embedding_model = SentenceTransformer(
            model_name_or_path=embedding_model_name,
            device=device
        )
        

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
    

    def query_and_summarize(
        self,
        query_text: str,
        top_k: int = 5,
        filter: Optional[Dict] = None,
        context_window: int = 1  # New parameter for context window
    ) -> Dict:
        """
        Query Pinecone and generate a summary with enhanced context handling.
        
        Args:
            query_text: Text to search for
            top_k: Number of results to return
            filter: Optional metadata filters
            context_window: Number of adjacent chunks to include for context
            
        Returns:
            Dictionary containing results, summary, and context information
        """
        # Get initial vector search results
        results = self.query(query_text, top_k, filter)
        
        # Expand context by including adjacent chunks
        expanded_results = []
        for match in results:
            page_num = match['metadata']['page_number']
            
            # Query for adjacent chunks on the same page
            context_filter = {
                "page_number": {
                    "$gte": max(1, page_num - context_window),
                    "$lte": page_num + context_window
                }
            }
            if filter:
                context_filter.update(filter)
            
            context_results = self.query(
                match['metadata']['text'],
                top_k=2 * context_window + 1,
                filter=context_filter
            )
            
            expanded_results.extend(context_results)
        
        # Remove duplicates while preserving order
        seen_ids = set()
        unique_results = []
        for result in expanded_results:
            if result['id'] not in seen_ids:
                seen_ids.add(result['id'])
                unique_results.append(result)
        
        # Sort by page number and position
        unique_results.sort(key=lambda x: (x['metadata']['page_number']))
        
        # Convert to LangChain documents with enhanced metadata
        documents = []
        for i, match in enumerate(unique_results):
            # Add contextual markers
            is_direct_match = any(r['id'] == match['id'] for r in results)
            context_type = "DIRECT MATCH" if is_direct_match else "CONTEXT"
            
            # Combine text with previous chunk if available
            current_text = match['metadata']['text']
            if i > 0 and unique_results[i-1]['metadata']['page_number'] == match['metadata']['page_number']:
                current_text = f"{unique_results[i-1]['metadata']['text']} {current_text}"
            
            documents.append(
                Document(
                    page_content=current_text,
                    metadata={
                        'score': match['score'],
                        'page_number': match['metadata']['page_number'],
                        'context_type': context_type,
                        'chunk_position': i
                    }
                )
            )
        
        # Generate summary using LangChain
        summary = self.chain.run(
            input_documents=documents,
            question=query_text
        )
        
        return {
            'raw_results': unique_results,
            'summary': summary,
            'source_pages': [doc.metadata['page_number'] for doc in documents],
            'context_info': {
                'direct_matches': len([d for d in documents if d.metadata['context_type'] == "DIRECT MATCH"]),
                'context_chunks': len([d for d in documents if d.metadata['context_type'] == "CONTEXT"]),
            }
        }



        


def main():
    # Initialize pipeline
    pipeline = PineconePipelineBetter(
        pinecone_api_key="pcsk_43sajZ_MjcXR2yN5cAcVi8RARyB6i3NP3wLTnTLugbUcN9cUU4q5EfNmuwLPkmxAvykk9o",
        google_api_key="AIzaSyB75kXU5OSZBp4oycOnjk-uavpJgg537yI" ,

        environment="us-east-1",  # Use the appropriate AWS region
        index_name="pdf-embeddings"
    )
    
    
    # Example query
    results = pipeline.query_and_summarize(
    query_text="Under section 42, the dissolution of the firm depends",
    top_k=5,
    context_window=1  # Adjust based on your needs
    )

    print(results['summary'])
    print(f"Sources from pages: {results['source_pages']}")
    print(f"Context stats: {results['context_info']}")

    
  

if __name__ == "__main__":
    main()