
import gradio as gr
from typing import List, Dict, Any, Optional
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain_community.vectorstores import Neo4jVector
from langchain.docstore.document import Document
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import networkx as nx
import plotly.graph_objects as go
import hashlib

class EnhancedLegalSearchSystem:
    def __init__(
        self,
        google_api_key: str,
        neo4j_url: str,
        neo4j_username: str,
        neo4j_password: str,
        embedding_model_name: str = "intfloat/e5-small-v2",
        device: str = "cpu"
    ):
        """Initialize the Enhanced Legal Search System"""
        # Initialize LLM
        self.llm = GoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=google_api_key,
            temperature=0.1
        )
        
        # Initialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key,
            task_type="retrieval_query"
        )
        
        # Initialize Neo4j connection
        self.neo4j_driver = GraphDatabase.driver(
            neo4j_url,
            auth=(neo4j_username, neo4j_password)
        )
        
        # Initialize vector store
        self.vector_store = Neo4jVector.from_existing_graph(
            embedding=self.embeddings,
            url=neo4j_url,
            username=neo4j_username,
            password=neo4j_password,
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding"
        )
        
        # Initialize additional embedding model for enhanced search
        self.local_embedding_model = SentenceTransformer(
            model_name_or_path=embedding_model_name,
            device=device
        )
        
        # Initialize prompts
        self.init_prompts()

    def __del__(self):
        """Cleanup Neo4j connection"""
        if hasattr(self, 'neo4j_driver'):
            self.neo4j_driver.close()

    def init_prompts(self):
        """Initialize enhanced prompts for legal analysis"""
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a legal expert assistant specializing in Indian law. 
                         Analyze the following legal context and provide a detailed, structured answer to the question.
                         Include specific sections, rules, and precedents where applicable.
                         Format your response with clear headings and bullet points for better readability.
                         
                         Context: {context}"""),
            ("human", "Question: {question}")
        ])

        self.map_prompt = PromptTemplate(
            template="""
            Analyze the following legal text segment:
            
            TEXT: "{text}"
            
            Instructions:
            1. Extract and summarize the key legal points
            2. Maintain all legal terminology exactly as written
            3. Preserve section numbers and references
            4. Keep all specific conditions and requirements
            5. Include any mentioned time periods or deadlines
            
            KEY POINTS:
            """,
            input_variables=["text"]  # Removed page_number as it's not used in the template
        )

        self.combine_prompt = PromptTemplate(
            template="""
            Question: {question}

            Using ONLY the information from the following legal document excerpts, provide a comprehensive answer:

            {text}

            Instructions:
            1. Base your response EXCLUSIVELY on the provided document excerpts
            2. If the documents don't contain enough information to fully answer the question, explicitly state what's missing
            3. Use direct quotes when appropriate
            4. Organize the response by relevant sections found in the documents
            5. If there are conflicting statements across documents, highlight them

            ANALYSIS:
            """,
            input_variables=["text", "question"]
        )

        # Initialize summarize chain
        self.chain = load_summarize_chain(
            llm=self.llm,
            chain_type="map_reduce",
            map_prompt=self.map_prompt,
            combine_prompt=self.combine_prompt,
            verbose=True
        )

   
    def get_related_legal_entities(self, query: str) -> List[Dict]:
        """Retrieve related legal entities and their relationships"""
        # Corrected Cypher query to handle aggregation properly
        cypher_query = """
                    // First, let's check if nodes exist and get their labels
MATCH (d:Document)
WHERE toLower(d.text) CONTAINS toLower($query)
WITH d
// Match all relationships from the document, collecting their types
OPTIONAL MATCH (d)-[r]-(connected)
WHERE NOT connected:Document  // Avoid direct document-to-document relations
WITH d, 
     collect(DISTINCT type(r)) as relationTypes,
     collect(DISTINCT labels(connected)) as connectedLabels

// Now use these to build our main query
MATCH (d:Document)-[r1]-(e)
WHERE toLower(d.text) CONTAINS toLower($query)
  AND NOT e:Document  // Exclude direct document connections
WITH d, r1, e
// Get secondary connections, but be more specific about what we're looking for
OPTIONAL MATCH (e)-[r2]-(related)
WHERE (related:Entity OR related:Concept OR related:Section OR related:Case)
  AND related <> d  // Prevent cycles back to original document
WITH d, {
    source_id: id(d),
    source_text: d.text,
    document_type: COALESCE(d.type, "Unknown"),
    relationship_type: type(r1),
    entity: {
        id: id(e),
        type: CASE WHEN e:Entity THEN "Entity"
                  WHEN e:Concept THEN "Concept"
                  WHEN e:Section THEN "Section"
                  WHEN e:Case THEN "Case"
                  ELSE "Other" END,
        text: COALESCE(e.text, e.name, e.title, "Unnamed"),
        properties: properties(e)
    },
    related_entities: collect(DISTINCT {
        id: id(related),
        type: CASE WHEN related:Entity THEN "Entity"
                  WHEN related:Concept THEN "Concept"
                  WHEN related:Section THEN "Section"
                  WHEN related:Case THEN "Case"
                  ELSE "Other" END,
        relationship: type(r2),
        text: COALESCE(related.text, related.name, related.title, "Unnamed"),
        properties: properties(related)
    })
} as result
WHERE result.entity.text IS NOT NULL  // Filter out any results with null entity text
RETURN DISTINCT result
ORDER BY result.source_id, result.entity.id
LIMIT 25

        """
        try:
            with self.neo4j_driver.session() as session:
                # Execute the improved query
                result = session.run(cypher_query, {"query": query})
                entities = [record["result"] for record in result]
                
                # Log the results for debugging
                print(f"Found {len(entities)} related entities")
                if entities:
                    for entity in entities:
                        print(f"Entity: {entity['entity']['text']}")
                        print(f"Source: {entity['source_text'][:100]}...")
                        print(f"Related: {len(entity['related_entities'])} connections")
                        
                return entities
                    
        except Exception as e:
            print(f"Error in get_related_legal_entities: {str(e)}")
            return []
        
    async def process_legal_query(
        self,
        question: str,
        top_k: int = 5,
        context_window: int = 1
    ) -> Dict[str, Any]:
        """Process a legal query using both graph and vector search capabilities"""
        try:
            # 1. Perform semantic search
            semantic_results = self.vector_store.similarity_search(
                question,
                k=top_k,
                search_type="hybrid"
            )
            
            # 2. Get related legal entities with the full question context
            related_entities = self.get_related_legal_entities(question)
            
            # Log the counts for debugging
            print(f"Found {len(semantic_results)} semantic results")
            print(f"Found {len(related_entities)} related entities")
            
            # 3. Expand context with related documents
            expanded_results = self.expand_context(
                semantic_results,
                context_window
            )
            
            # 4. Generate comprehensive answer
            documents = self._process_results(expanded_results, semantic_results)
            
            # 5. Prepare context for LLM
            context = self._prepare_context(documents, related_entities)
            
            # 6. Generate answer using LLM
            chain = LLMChain(llm=self.llm, prompt=self.qa_prompt)
            response = await chain.ainvoke({
                "context": context,
                "question": question
            })
            answer = response.get('text', '')
            
            # 7. Return structured response with explicit related concepts
            return {
                "status": "Success",
                "answer": answer,
                "documents": self._format_documents(documents),
                "related_concepts": related_entities,  # This should now contain data
                "source_ids": sorted(list(set(doc.metadata.get('document_id', 'unknown') for doc in documents))),
                "context_info": {
                    "direct_matches": len([d for d in documents if d.metadata.get('context_type') == "DIRECT MATCH"]),
                    "context_chunks": len([d for d in documents if d.metadata.get('context_type') == "CONTEXT"])
                }
            }
            
        except Exception as e:
            print(f"Error in process_legal_query: {str(e)}")  # Add error logging
            return {
                "status": f"Error: {str(e)}",
                "answer": "An error occurred while processing your query.",
                "documents": "",
                "related_concepts": [],
                "source_ids": [],
                "context_info": {}
            }


    def expand_context(
        self,
        initial_results: List[Document],
        context_window: int
    ) -> List[Document]:
        """Expand context around search results"""
        expanded_results = []
        seen_ids = set()
        
        for doc in initial_results:
            doc_id = doc.metadata.get('document_id', doc.page_content[:50])
            if doc_id not in seen_ids:
                # Query for related documents
                context_results = self.vector_store.similarity_search(
                    doc.page_content,
                    k=2 * context_window + 1,
                    search_type="hybrid"
                )
                
                for result in context_results:
                    result_id = result.metadata.get('document_id', result.page_content[:50])
                    if result_id not in seen_ids:
                        expanded_results.append(result)
                        seen_ids.add(result_id)
        
        return expanded_results

    def _process_results(self, expanded_results: List[Document], initial_results: List[Document]) -> List[Document]:
        """Process and deduplicate search results"""
        seen_ids = set()
        documents = []
        
        for doc in expanded_results:
            doc_id = doc.metadata.get('document_id', doc.page_content[:50])
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                is_direct_match = any(
                    r.metadata.get('document_id', r.page_content[:50]) == doc_id
                    for r in initial_results
                )
                
                doc.metadata['context_type'] = (
                    "DIRECT MATCH" if is_direct_match else "CONTEXT"
                )
                documents.append(doc)
        
        return sorted(
            documents,
            key=lambda x: x.metadata.get('document_id', 'unknown')
        )

    def _prepare_context(
        self,
        documents: List[Document],
        related_entities: List[Dict]
    ) -> str:
        """Prepare context for LLM processing"""
        context = "\n\nLegal Documents:\n" + "\n".join([
            f"[Document ID: {doc.metadata.get('document_id', 'unknown')}] {doc.page_content}"
            for doc in documents
        ])
        
        if related_entities:
            context += "\n\nRelated Legal Concepts and Relationships:\n"
            for entity in related_entities:
                context += f"\n• {entity.get('entity', '')}"
                if entity.get('related_entities'):
                    for related in entity['related_entities']:
                        if related.get('entity'):
                            context += f"\n  - {related['type']}: {related['entity']}"
        
        return context

    def _format_documents(self, documents: List[Document]) -> str:
        """Format documents as markdown"""
        markdown = "### Retrieved Documents\n\n"
        for i, doc in enumerate(documents, 1):
            markdown += (
                f"**Document {i}** "
                f"(ID: {doc.metadata.get('document_id', 'unknown')}, "
                f"{doc.metadata.get('context_type', 'UNKNOWN')})\n"
                f"```\n{doc.page_content}\n```\n\n"
            )
        return markdown

    

    def generate_document_graph(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.5
    ) -> List[Dict]:
        """Generate graph data based on document similarity and relationships"""
        try:
            # 1. Get initial semantic search results
            semantic_results = self.vector_store.similarity_search(
                query,
                k=top_k,
                search_type="hybrid"
            )
            
            # 2. Get embeddings for all documents
            doc_texts = [doc.page_content for doc in semantic_results]
            doc_embeddings = self.local_embedding_model.encode(doc_texts)
            
            # 3. Create graph data structure
            graph_data = []
            seen_docs = set()
            
            # First, add all documents as nodes
            for i, doc in enumerate(semantic_results):
                doc_id = doc.metadata.get('document_id', f'doc_{i}')
                if doc_id not in seen_docs:
                    seen_docs.add(doc_id)
                    doc_type = doc.metadata.get('type', 'document')
                    
                    # Create node entry
                    graph_data.append({
                        'source_id': doc_id,
                        'source_text': doc.page_content[:200],  # Truncate for display
                        'document_type': doc_type,
                        'entity': {
                            'id': doc_id,
                            'type': 'Document',
                            'text': f"Document {i + 1}",
                            'properties': {
                                'similarity': 1.0,
                                'length': len(doc.page_content)
                            }
                        },
                        'related_entities': []
                    })
            
            # Add relationships based on similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarity_matrix = cosine_similarity(doc_embeddings)
            
            # Create relationships between similar documents
            for i in range(len(semantic_results)):
                related = []
                for j in range(len(semantic_results)):
                    if i != j and similarity_matrix[i][j] > similarity_threshold:
                        doc_j = semantic_results[j]
                        doc_j_id = doc_j.metadata.get('document_id', f'doc_{j}')
                        
                        related.append({
                            'id': doc_j_id,
                            'type': 'Document',
                            'relationship': 'similar_to',
                            'text': f"Document {j + 1}",
                            'properties': {
                                'similarity_score': float(similarity_matrix[i][j])
                            }
                        })
                
                # Add related documents to the graph data
                if related:
                    graph_data[i]['related_entities'] = related
            
            return graph_data
            
        except Exception as e:
            print(f"Error generating document graph: {str(e)}")
            return []



def create_graph_visualization(graph_data: List[Dict]) -> go.Figure:
    """Create an interactive graph visualization using Plotly"""
    if not graph_data:
        return go.Figure(layout=go.Layout(title='No documents found'))
    
    # Initialize graph
    G = nx.Graph()
    
    # Color mapping
    color_map = {
        'Document': '#3B82F6',  # blue
        'Section': '#10B981',   # green
        'Reference': '#F59E0B'  # yellow
    }
    
    # Node information storage
    node_colors = []
    node_texts = []
    node_hovers = []  # Full text for hover
    nodes_added = set()
    
    # Process nodes and edges
    for data in graph_data:
        source_id = data['source_id']
        source_text = data['source_text']
        
        # Add main document node
        if source_id not in nodes_added:
            G.add_node(source_id)
            node_colors.append(color_map['Document'])
            # Short text for display
            node_texts.append(f"Doc {len(nodes_added)+1}")
            # Full text for hover/click
            node_hovers.append(f"Document {len(nodes_added)+1}:<br><br>{source_text}")
            nodes_added.add(source_id)
        
        # Process related documents
        for related in data.get('related_entities', []):
            related_id = related['id']
            similarity = related['properties'].get('similarity_score', 0.0)
            
            if related_id not in nodes_added:
                G.add_node(related_id)
                node_colors.append(color_map['Document'])
                node_texts.append(f"Doc {len(nodes_added)+1}")
                node_hovers.append(f"Document {len(nodes_added)+1}:<br><br>{related['text']}")
                nodes_added.add(related_id)
            
            # Add edge with similarity weight
            G.add_edge(
                source_id,
                related_id,
                weight=similarity,
                relationship=f"Similarity: {similarity:.2f}"
            )
    
    # Create layout
    pos = nx.spring_layout(G, k=2.0, iterations=50)
    
    # Create edge trace
    edge_x = []
    edge_y = []
    edge_text = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        # Create curved line
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        # Add some curvature
        mid_x += (y1 - y0) * 0.1
        mid_y -= (x1 - x0) * 0.1
        
        # Add points for curved line
        edge_x.extend([x0, mid_x, x1, None])
        edge_y.extend([y0, mid_y, y1, None])
        edge_text.append(edge[2]['relationship'])
    
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1.5, color='#9CA3AF'),
        hoverinfo='text',
        text=edge_text,
        mode='lines'
    )
    
    # Create node trace
    node_x = []
    node_y = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
    
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_texts,
        hovertext=node_hovers,  # Full text shown on hover
        textposition="top center",
        marker=dict(
            size=30,
            color=node_colors,
            line=dict(width=2, color='white'),
            symbol='circle'
        ),
        customdata=node_hovers  # Store full text for click events
    )
    
    # Create figure with updated layout
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title={
                'text': 'Document Similarity Graph<br><sub>Click nodes to view full text</sub>',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=60),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            width=800,
            height=600,
            clickmode='event+select'  # Enable click events
        )
    )
    
    return fig

def create_interface(search_system: EnhancedLegalSearchSystem):
    """Create Gradio interface with interactive graph"""
    
    with gr.Blocks(css="footer {display: none !important;}") as demo:
        gr.Markdown("""
        # Enhanced Legal Search System
        Enter your legal query below to search through documents and get an AI-powered analysis.
        This system combines graph-based and semantic search capabilities for comprehensive legal research.
        """)
        
        with gr.Row():
            query_input = gr.Textbox(
                label="Legal Query",
                placeholder="e.g., What are the reporting obligations for banks under the Money Laundering Act?",
                lines=3
            )
            
        with gr.Row():
            search_button = gr.Button("Search & Analyze")
            
        status_output = gr.Textbox(
            label="Status",
            interactive=False
        )
        
        with gr.Tabs():
            with gr.TabItem("AI Legal Analysis"):
                analysis_output = gr.Markdown(
                    label="AI-Generated Legal Analysis",
                    value="Analysis will appear here..."
                )
            
            with gr.TabItem("Retrieved Documents"):
                docs_output = gr.Markdown(
                    label="Source Documents",
                    value="Search results will appear here..."
                )
                
            with gr.TabItem("Related Concepts"):
                concepts_output = gr.Json(
                    label="Related Legal Concepts",
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
        
        async def process_query(query):
            if not query.strip():
                return (
                    "Please enter a query",
                    "No analysis available",
                    "No documents available",
                    {},
                    None,
                    ""
                )
            
            results = await search_system.process_legal_query(query)
            graph_data = search_system.generate_document_graph(query)
            graph_fig = create_graph_visualization(graph_data)
            
            return (
                results['status'],
                results['answer'],
                results['documents'],
                {"related_concepts": results['related_concepts']},
                graph_fig,
                "Click on a node to view document content"
            )
        
        search_button.click(
            fn=process_query,
            inputs=[query_input],
            outputs=[
                status_output,
                analysis_output,
                docs_output,
                concepts_output,
                graph_output,
                selected_doc_content
            ]
        )
    
    return demo



if __name__ == "__main__":
    search_system = EnhancedLegalSearchSystem(
        google_api_key="AIzaSyB75kXU5OSZBp4oycOnjk-uavpJgg537yI",
        neo4j_url="neo4j+s://ffc2cc0f.databases.neo4j.io",
        neo4j_username="neo4j",
        neo4j_password="iH1Qe61EwRwhWtoVncW4XiADuUaABOvKtOagu1NY1m4"
    )
    demo = create_interface(search_system)
    demo.launch()


