import os
import gradio as gr
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
import time
from graphGradio import EnhancedLegalSearchSystem
import asyncio
from graphGradio import create_graph_visualization


class LegalSearchSystem:
    def __init__(
        self,
        pinecone_api_key: str = "pcsk_43sajZ_MjcXR2yN5cAcVi8RARyB6i3NP3wLTnTLugbUcN9cUU4q5EfNmuwLPkmxAvykk9o",
        google_api_key: str = "AIzaSyB75kXU5OSZBp4oycOnjk-uavpJgg537yI",
        environment: str = "us-east-1",
        index_name: str = "pdf-embeddings",
        dimension: int = 384,
        embedding_model_name: str = "intfloat/e5-small-v2",
        device: str = "cpu"
    ):
        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        
        # Initialize LangChain with Gemini
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0,
            google_api_key=google_api_key
        )

        # Initialize prompts
        self.map_prompt = PromptTemplate(
            template="""
            Analyze the following legal text segment and extract key information:
            
            TEXT: "{text}"
            
            Instructions:
            1. Maintain all legal terminology exactly as written
            2. Preserve section numbers and references
            3. Keep all specific conditions and requirements
            4. Include any mentioned time periods or deadlines
            
            DETAILED ANALYSIS:
            """,
            input_variables=["text"]
        )
        
        self.combine_prompt = PromptTemplate(
            template="""
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
            """,
            input_variables=["text", "question"]
        )
        
        # Initialize chain
        self.chain = load_summarize_chain(
            llm=self.llm,
            chain_type="stuff",
            prompt=self.combine_prompt,
            verbose=True
        )

        # Initialize Pinecone index and embedding model
        self.index = self.pc.Index(index_name)
        self.embedding_model = SentenceTransformer(
            model_name_or_path=embedding_model_name,
            device=device
        )

    def search(self, query_text: str, top_k: int = 5, context_window: int = 1) -> Dict:
        """
        Perform a search and analysis of the legal query.
        """
        try:
            # Get search results with context
            results = self.query_and_summarize(
                query_text=query_text,
                top_k=top_k,
                context_window=context_window
            )
            
            # Format the results for display
            docs_markdown = self._format_documents(results['raw_results'])
            
            return {
                'status': "Search completed successfully",
                'documents': docs_markdown,
                'analysis': results['summary'],
                'source_pages': results['source_pages'],
                'context_info': results['context_info']
            }
        except Exception as e:
            return {
                'status': f"Error during search: {str(e)}",
                'documents': "Error retrieving documents",
                'analysis': "Error generating analysis",
                'source_pages': [],
                'context_info': {}
            }

    def query_and_summarize(
        self,
        query_text: str,
        top_k: int = 5,
        filter: Optional[Dict] = None,
        context_window: int = 1
    ) -> Dict:
        """
        Query Pinecone and generate a summary with enhanced context handling.
        """
        # Generate embedding for query
        query_embedding = self.embedding_model.encode(query_text).tolist()
        
        # Query Pinecone
        initial_results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter
        )['matches']
        
        # Expand context
        expanded_results = []
        for match in initial_results:
            page_num = match['metadata']['page_number']
            context_filter = {
                "page_number": {
                    "$gte": max(1, page_num - context_window),
                    "$lte": page_num + context_window
                }
            }
            if filter:
                context_filter.update(filter)
            
            context_results = self.index.query(
                vector=self.embedding_model.encode(match['metadata']['text']).tolist(),
                top_k=2 * context_window + 1,
                include_metadata=True,
                filter=context_filter
            )['matches']
            
            expanded_results.extend(context_results)
        
        # Process results and generate summary
        documents = self._process_results(expanded_results, initial_results)
        summary = self.chain.run(
            input_documents=documents,
            question=query_text
        )
        
        return {
            'raw_results': expanded_results,
            'summary': summary,
            'source_pages': list(set(doc.metadata['page_number'] for doc in documents)),
            'context_info': {
                'direct_matches': len([d for d in documents if d.metadata['context_type'] == "DIRECT MATCH"]),
                'context_chunks': len([d for d in documents if d.metadata['context_type'] == "CONTEXT"])
            }
        }

    def _process_results(self, expanded_results: List[Dict], initial_results: List[Dict]) -> List[Document]:
        """
        Process and deduplicate search results.
        """
        seen_ids = set()
        documents = []
        
        for result in expanded_results:
            if result['id'] not in seen_ids:
                seen_ids.add(result['id'])
                is_direct_match = any(r['id'] == result['id'] for r in initial_results)
                
                documents.append(Document(
                    page_content=result['metadata']['text'],
                    metadata={
                        'score': result['score'],
                        'page_number': result['metadata']['page_number'],
                        'context_type': "DIRECT MATCH" if is_direct_match else "CONTEXT"
                    }
                ))
        
        return sorted(documents, key=lambda x: x.metadata['page_number'])

    def _format_documents(self, results: List[Dict]) -> str:
        """
        Format search results as markdown.
        """
        markdown = "### Retrieved Documents\n\n"
        for i, result in enumerate(results, 1):
            markdown += f"**Document {i}** (Page {result['metadata']['page_number']})\n"
            markdown += f"```\n{result['metadata']['text']}\n```\n\n"
        return markdown
    

async def process_query_async(query: str, search_system: LegalSearchSystem, graph_search_system: EnhancedLegalSearchSystem):
    """
    Asynchronous function to process both traditional and graph-based searches
    """
    if not query.strip():
        return "Please enter a query", "", "", "", {}
    
    # Regular search (synchronous)
    results = search_system.search(query)
    
    try:
        # Graph search (asynchronous)
        graph_results = await graph_search_system.process_legal_query(query)
        graph_documents = graph_results.get('documents', "Error processing graph search")
        graph_concepts = graph_results.get('related_concepts', {})
    except Exception as e:
        graph_documents = f"Error processing graph search: {str(e)}"
        graph_concepts = {}

    graph_data = graph_search_system.generate_document_graph(query)
    graph_fig = create_graph_visualization(graph_data)
    
    return (
        results['status'],
        results['documents'],
        results['analysis'],
        graph_documents,
        graph_concepts,
        graph_fig,
        "Click on a node to view document content"
    )

def create_interface(graph_search_system: EnhancedLegalSearchSystem):
    search_system = LegalSearchSystem()
    
    with gr.Blocks(css="footer {display: none !important;}") as demo:
        gr.Markdown("""
        # Legal Search AI with LangChain
        Enter your legal query below to search through documents and get an AI-powered analysis.
        """)
        
        with gr.Row():
            query_input = gr.Textbox(
                label="Legal Query",
                placeholder="e.g., What are the key principles of contract law?",
                lines=3
            )
            
        with gr.Row():
            search_button = gr.Button("Search & Analyze")
            
        status_output = gr.Textbox(
            label="Status",
            interactive=False
        )
        
        with gr.Tabs():
            with gr.TabItem("Search Results"):
                docs_output = gr.Markdown(
                    label="Retrieved Documents",
                    value="Search results will appear here..."
                )
            
            with gr.TabItem("AI Legal Analysis"):
                summary_output = gr.Markdown(
                    label="AI-Generated Legal Analysis",
                    value="Analysis will appear here..."
                )

            with gr.TabItem("Retrieved Documents through Graph Rag"):
                docs_output_graph = gr.Markdown(
                    label="Source Documents",
                    value="Search results will appear here..."
                )
                graph_analysis_output = gr.JSON(
                    label="Related Concepts",
                    value={}
                )

            with gr.TabItem("Knowledge Graph"):
                # Graph visualization
                graph_output = gr.Plot(
                    label="Legal Knowledge Graph"
                )
                # Add text area for showing clicked document content
                selected_doc_content = gr.Textbox(
                    label="Selected Document Content",
                    interactive=False,
                    lines=10
                )

        def process_query(query):
        # Create event loop if it doesn't exist
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the async function and get results
            return loop.run_until_complete(
                process_query_async(query, search_system, graph_search_system)
            )
    
        search_button.click(
            fn=process_query,
            inputs=[query_input],
            outputs=[
                status_output,
                docs_output,
                summary_output,
                docs_output_graph,
                graph_analysis_output,
                graph_output,
                selected_doc_content
            ]
        )
    
    
    
    return demo

if __name__ == "__main__":
    graph_search_system = EnhancedLegalSearchSystem(
        google_api_key="AIzaSyB75kXU5OSZBp4oycOnjk-uavpJgg537yI",
        neo4j_url="neo4j+s://ffc2cc0f.databases.neo4j.io",
        neo4j_username="neo4j",
        neo4j_password="iH1Qe61EwRwhWtoVncW4XiADuUaABOvKtOagu1NY1m4"    
    )
    demo = create_interface(graph_search_system)
    demo.launch()