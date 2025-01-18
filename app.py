from flask import Flask, request, jsonify
from database import PineconePipeline

app = Flask(__name__)

@app.route("/")
def hello():
    return "hello"

@app.route('/query', methods=['POST'])
def relevant_documents(): 
    # Get JSON data from the request
    data = request.json

    # Retrieve query parameters from the JSON data
    query_text = data.get("query_text")
    top_k = data.get("top_k", 5)  # Default to 5 if not provided
    filters = data.get("filters", {})

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

    return jsonify(response), 200

if __name__ == "__main__":
    app.run(debug=True)
