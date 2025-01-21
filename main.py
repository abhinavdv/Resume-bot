from flask import Flask, render_template, request, Response, stream_with_context, jsonify
import json
import os
import vertexai
from vertexai.preview.generative_models import GenerativeModel, SafetySetting, Part, Tool
from vertexai.preview.generative_models import grounding

app = Flask(__name__)
print("abc 2")
# Initialize Vertex AI
vertexai.init(project="concise-option-448310-n8", location="us-central1")

# Configure generation settings
generation_config1 = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}
generation_config2 = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
    "response_mime_type": "application/json",
}


# Configure safety settings
safety_settings = [
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
]

# System instruction for the model
SYSTEM_INSTRUCTION = """You are an AI bot with context of my resume. Try to answer questions that you even know paritally about.
If you dont know the completeanswer, ask the user to contact me directly at my email: dvabhinav31@gmail.com or reach out to me on linkedin: https://www.linkedin.com/in/abhinav-duvvuri/"""

def get_vertex_model():
    tools = [
        Tool.from_retrieval(
            retrieval=grounding.Retrieval(
                source=grounding.VertexAISearch(
                    datastore="projects/concise-option-448310-n8/locations/global/collections/default_collection/dataStores/resume-portfolio_1737284302495"
                ),
            )
        ),
    ]
    return GenerativeModel(
        "gemini-1.5-flash-002",
        tools=tools,
        system_instruction=[SYSTEM_INSTRUCTION]
    )

def generate_suggested_questions():
    model = get_vertex_model()
    chat = model.start_chat()
    try:
        response = chat.send_message(
            """Generate 5 focused questions that highlight my most impressive qualifications and experience. Keep the questions very short and concise. Ask only questions that can be answered from the provided document.
            Return ONLY the questions in a JSON array format.""",
            generation_config=generation_config2,
            safety_settings=safety_settings
        )
        
        # Parse the response to get just the text content
        response_text = ""
        if hasattr(response, 'candidates') and response.candidates:
            for candidate in response.candidates:
                if hasattr(candidate, 'content') and candidate.content:
                    for part in candidate.content.parts:
                        if hasattr(part, 'text'):
                            response_text = part.text
                            break
        
        # Try to parse the text as JSON
        try:
            questions = json.loads(response_text)
            if isinstance(questions, list) and len(questions) > 0:
                return questions
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract questions from text
            print(f"Failed to parse JSON, response was: {response_text}")
            print("response")
            print(response)
            
            # If all parsing fails, return default questions
            return [
                "What are Abhinav's key skills and expertise?",
                "What is Abhinav's most significant project?",
                "Why should I hire Abhinav?"
            ]
            
    except Exception as e:
        print(f"Error generating questions: {e}")
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
        try:
            model = get_vertex_model()
            chat = model.start_chat()
            
            response = chat.send_message(
                query,
                generation_config=generation_config1,
                safety_settings=safety_settings,
                stream=True  # Enable streaming
            ) 
            
            # Stream the response
            for chunk in response:
                if hasattr(chunk, 'candidates') and chunk.candidates:
                    for candidate in chunk.candidates:
                        if hasattr(candidate, 'content') and candidate.content:
                            for part in candidate.content.parts:
                                if hasattr(part, 'text'):
                                    yield "data: " + json.dumps({"content": part.text}) + "\n\n"
                
        except Exception as e:
            print(f"Error during response generation: {e}")
            yield "data: " + json.dumps({"content": "\nSorry, there was an error processing your request."}) + "\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Transfer-Encoding': 'chunked'
        }
    )

if __name__ == '__main__':
    app.run(debug=True,port=int(os.environ.get("PORT", 8080)),host='0.0.0.0')