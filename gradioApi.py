from gradioApi import api
from database import PineconePipeline

@api('/query')
def relevant_documents(query_text: str, top_k: int = 5, filters: dict = {}):
    # Initialize the PineconePipeline
    pipeline = PineconePipeline(
        api_key="pcsk_43sajZ_MjcXR2yN5cAcVi8RARyB6i3NP3wLTnTLugbUcN9cUU4q5EfNmuwLPkmxAvykk9o",
        environment="us-west-2",
        index_name="pdf-embeddings"
    )

    # Get the query results
    results = pipeline.query_and_get_results(
        query_text=query_text,
        top_k=top_k,
        filter=filters
    )

    # Format the response
    response = []
    for match in results:
        response.append({
            "score": match["score"],
            "text": match["metadata"]["text"],
            "page_number": match["metadata"]["page_number"]
        })

    return response
