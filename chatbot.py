# chatbot.py
from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

app = Flask(__name__)

# API Keys
supermemory_key = os.getenv('ARYAN_SUPERMEMORY_API_KEY')
gemini_key = os.getenv('ARYAN_GEMINI_KEY')  # Not used here, but kept for consistency
groq_key = os.getenv('groq_api_key')

# Supermemory client (for potential manual ops, but proxy handles auto)
from supermemory import Supermemory
client = Supermemory(api_key=supermemory_key)

# Groq client via Supermemory proxy (auto-handles memory storage/retrieval)
groq_client = OpenAI(
    api_key=groq_key,
    base_url="https://api.supermemory.ai/v3/https://api.groq.com/openai/v1",
    default_headers={
        "x-supermemory-api-key": supermemory_key,
        "x-sm-user-id": "user_123"  # Enables auto-memory scoping
    }
)

# Simplified LangChain Prompt Template for Research Assistant
# (Relies on proxy auto-injection for context; no manual memory var)
research_prompt = PromptTemplate(
    input_variables=["user_query"],
    template="""You are an expert research assistant powered by advanced AI. Your goal is to provide accurate, insightful, and well-structured responses to research-oriented queries. 
Draw from our shared conversation history and knowledge to personalize and enhance your assistance.

User Query: {user_query}

dont use any special symbols especially *, |, etc. answer in pure texts
"""
)

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message')
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        # Format LangChain prompt (proxy auto-injects memory behind the scenes)
        system_prompt = research_prompt.format(
            user_query=user_message
        )

        # Build messages with system prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        # Call Groq via proxy (auto-handles memory injection on top of our prompt)
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",  # Or swap to llama3-70b-8192 for faster research tasks
            messages=messages,
            max_tokens=1000,
            temperature=0.7  # Balanced for research: creative yet factual
        )

        ai_response = response.choices[0].message.content

        # Proxy auto-stores exchanges, so no manual add needed

        return jsonify({'response': ai_response})

    except Exception as error:
        return jsonify({'error': str(error)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)