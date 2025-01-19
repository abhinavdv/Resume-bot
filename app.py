from flask import Flask, render_template, request, Response, stream_with_context, jsonify
from openai import OpenAI
import chromadb
import json
import keys

app = Flask(__name__)

# Initialize ChromaDB and OpenAI
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="resume_data", get_or_create=True)
openai_client = OpenAI(api_key=keys.OPENAI_API_KEY)

# Read and add documents to ChromaDB
import os

# Example: reading text files from a directory
documents = []
metadatas = []
ids = []

doc_dir = "docs"
for i, filename in enumerate(os.listdir(doc_dir)):
    if filename.endswith(".txt"):  # or any other extension you're using
        with open(os.path.join(doc_dir, filename), 'r') as file:
            content = file.read()
            documents.append(content)
            metadatas.append({"source": filename})
            ids.append(f"doc_{i}")

# Add documents to the collection
collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

def get_relevant_context(query, k=3):
    results = collection.query(
        query_texts=[query],
        n_results=k 
    )
    context = "\n\n".join(results['documents'][0])
    return context

def generate_suggested_questions():
    results = collection.query(
        query_texts=["Tell me everything about Abhinav"],
        n_results=10
    )
    context = "\n\n".join(results['documents'][0])
    
    completion = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant tasked with generating engaging questions about Abhinav's resume. Generate 3 diverse and impactful questions that would help someone understand Abhinav's key strengths and experience. Return ONLY the questions in a JSON array format."
            },
            {
                "role": "system",
                "content": f"Here's the resume content to base questions on: {context}"
            },
            {
                "role": "user",
                "content": "Generate 3 focused questions that highlight Abhinav's most impressive qualifications and experience."
            }
        ]
    )
    
    try:
        questions = json.loads(completion.choices[0].message.content)
        return questions
    except json.JSONDecodeError:
        # Reduced fallback questions to 3
        return [
            "What are Abhinav's key skills and expertise?",
            "What is Abhinav's most significant project?",
            "Why should I hire Abhinav?"
        ]

@app.route('/')
def home():
    suggested_questions = generate_suggested_questions()
    return render_template('index.html', suggested_questions=suggested_questions)

@app.route('/refresh-questions', methods=['GET'])
def refresh_questions():
    suggested_questions = generate_suggested_questions()
    return jsonify({"questions": suggested_questions})

@app.route('/ask', methods=['GET'])
def ask():
    query = request.args.get('question')
    
    def generate():
        if query.lower() == "bye":
            yield "data: " + json.dumps({"content": "Have a lovely day!"}) + "\n\n"
            return
        
        context = get_relevant_context(query)
        completion = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an assistant who talks about Abhinav Duvvuri's resume. Only use the provided context to answer questions. If you can't find the information in the context, say you don't have that information. Keep the answers short and succinct."},
                {"role": "system", "content": f"Context from resume: {context}"},
                {"role": "user", "content": query}
            ],
            stream=True
        )
        
        for chunk in completion:
            if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content is not None:
                yield "data: " + json.dumps({"content": chunk.choices[0].delta.content}) + "\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Transfer-Encoding': 'chunked'
        }
    )

if __name__ == '__main__':
    app.run(debug=True) 