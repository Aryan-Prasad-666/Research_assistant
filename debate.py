from typing import Dict, TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_cohere import ChatCohere
from dotenv import load_dotenv
from functools import lru_cache
import os
import logging
import json
from datetime import datetime
import re
from flask import Flask, request, render_template, send_file, jsonify
import io

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()

# Validate environment variables
groq_key = os.getenv('groq2_api_key')
gemini_key = os.getenv('gemini_api_key')
cohere_key = os.getenv('cohere_api_key')
if not all([groq_key, gemini_key, cohere_key]):
    raise ValueError("Missing one or more API keys in environment variables")

# Initialize LLMs
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=groq_key,
    temperature=0.6
)
llm2 = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=gemini_key,
    temperature=0.6
)
llm3 = ChatCohere(
    api_key=cohere_key,
    temperature=0.6
)

def sanitize_text(text):
    """Replace or remove problematic Unicode characters."""
    if not isinstance(text, str):
        return text
    text = text.replace('\u2011', '-')
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text

def create_debate_workflow(debate_topic, max_turns=10):
    output_parser = CommaSeparatedListOutputParser()
    try:
        output = sanitize_text(llm.invoke("I wish to have a debate on {}. What would be the fighting sides called? Output just the names and nothing else as comma separated list".format(debate_topic)).content)
        classes = output_parser.parse(output)
        if len(classes) != 2:
            raise ValueError(f"Expected exactly two debate sides, got: {classes}")
        logger.info(f"Debate sides: {classes}")
    except Exception as e:
        logger.error(f"Error determining debate sides: {e}")
        raise

    class GraphState(TypedDict):
        classification: Optional[str]
        history: str
        current_response: Optional[str]
        count: int
        results: Optional[str]
        greeting: Optional[str]

    workflow = StateGraph(GraphState)

    prefix_start = (
        'You are in support of {}. You are in a debate with {} over the topic: {}. '
        'This is the conversation so far \n{}\n. '
        'Provide a unique, concise argument (one sentence) to support {}, '
        'countering {} with a specific real-world example, statistic, or reasoning, '
        'and avoid repeating prior arguments.'
    )

    @lru_cache(maxsize=100)
    def classify(question, class0, class1):
        try:
            result = sanitize_text(llm2.invoke("classify the sentiment of input as {} or {}. Output just the class. Input:{}".format(class0, class1, question)).content.strip())
            logger.info(f"Classified input '{question[:50]}...' as: {result}")
            return result
        except Exception as e:
            logger.error(f"Error classifying input: {e}")
            return class0

    def classify_input_node(state):
        question = state.get('current_response', '')
        classification = classify(question, '_'.join(classes[0].split(' ')), '_'.join(classes[1].split(' ')))
        return {"classification": classification}

    def handle_greeting_node(state):
        greeting = f"Hello! Today we will witness the fight between {classes[0]} vs {classes[1]}"
        logger.info(f"Generated greeting: {greeting}")
        return {"greeting": greeting}

    def handle_pro(state):
        try:
            summary = state.get('history', '').strip()
            current_response = state.get('current_response', '').strip()
            prompt = prefix_start.format(classes[0], classes[1], debate_topic, summary, classes[0], current_response or "Nothing")
            argument = classes[0] + ": " + sanitize_text(llm.invoke(prompt).content)
            logger.info(f"Pro argument: {argument}")
            return {"history": summary + '\n' + argument, "current_response": argument, "count": state.get('count', 0) + 1}
        except Exception as e:
            logger.error(f"Error in handle_pro: {e}")
            return {"history": summary + '\n' + f"{classes[0]}: Error generating argument", "current_response": "Error", "count": state.get('count', 0) + 1}

    def handle_opp(state):
        try:
            summary = state.get('history', '').strip()
            current_response = state.get('current_response', '').strip()
            prompt = prefix_start.format(classes[1], classes[0], debate_topic, summary, classes[1], current_response or "Nothing")
            argument = classes[1] + ": " + sanitize_text(llm2.invoke(prompt).content)
            logger.info(f"Opp argument: {argument}")
            return {"history": summary + '\n' + argument, "current_response": argument, "count": state.get('count', 0) + 1}
        except Exception as e:
            logger.error(f"Error in handle_opp: {e}")
            return {"history": summary + '\n' + f"{classes[1]}: Error generating argument", "current_response": "Error", "count": state.get('count', 0) + 1}

    def result(state):
        try:
            summary = state.get('history', '').strip()
            prompt = "Summarize the conversation and judge who won the debate. No ties are allowed. Conversation: {}".format(summary)
            result = sanitize_text(llm3.invoke(prompt).content)
            logger.info(f"Debate result: {result}")
            return {"results": result}
        except Exception as e:
            logger.error(f"Error in result: {e}")
            return {"results": "Error summarizing debate"}

    workflow.add_node("classify_input", classify_input_node)
    workflow.add_node("handle_greeting", handle_greeting_node)
    workflow.add_node("handle_pro", handle_pro)
    workflow.add_node("handle_opp", handle_opp)
    workflow.add_node("result", result)

    def decide_next_node(state):
        classification = state.get('classification')
        logger.debug(f"Deciding next node based on classification: {classification}")
        return "handle_opp" if classification == '_'.join(classes[0].split(' ')) else "handle_pro"

    def check_conv_length(state):
        count = state.get("count", 0)
        logger.debug(f"Conversation turn count: {count}/{max_turns}")
        return "result" if count >= max_turns else "classify_input"

    workflow.add_conditional_edges(
        "classify_input",
        decide_next_node,
        {"handle_pro": "handle_pro", "handle_opp": "handle_opp"}
    )
    workflow.add_conditional_edges(
        "handle_pro",
        check_conv_length,
        {"result": "result", "classify_input": "classify_input"}
    )
    workflow.add_conditional_edges(
        "handle_opp",
        check_conv_length,
        {"result": "result", "classify_input": "classify_input"}
    )

    workflow.set_entry_point("handle_greeting")
    workflow.add_edge('handle_greeting', "handle_pro")
    workflow.add_edge('result', END)

    return workflow, classes

@app.route('/multi-agent-debate', methods=['GET', 'POST'])
def multi_agent_debate():
    if request.method == 'POST':
        try:
            debate_topic = request.form.get('debate_topic', '').strip()
            max_turns = int(request.form.get('max_turns', 10))
            if not debate_topic:
                return render_template('multi_agent_debate.html', error="Debate topic is required", generation_date=datetime.now())

            # Create and run the debate workflow
            workflow, classes = create_debate_workflow(debate_topic=debate_topic, max_turns=max_turns)
            app_workflow = workflow.compile()
            conversation = app_workflow.invoke({'count': 0, 'history': '', 'current_response': ''})

            # Parse history into turns
            history = conversation.get('history', 'No conversation history generated')
            history_turns = []
            current_turn = None
            for line in history.split("\n"):
                line = line.strip()
                if line.startswith(f"{classes[0]}:") or line.startswith(f"{classes[1]}:"):
                    if current_turn:
                        history_turns.append(current_turn)
                    current_turn = line
                elif line and current_turn:
                    current_turn += "\n" + line
            if current_turn:
                history_turns.append(current_turn)

            # Prepare conversation data
            conversation_data = {
                'greeting': conversation.get('greeting', 'No greeting generated'),
                'history': history_turns,
                'result': conversation.get('results', 'No result generated')
            }

            # Save debate data for downloads
            global last_debate_data
            last_debate_data = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "debate_topic": debate_topic,
                    "max_turns": max_turns,
                    "debate_sides": classes,
                    "models": {
                        "pro_arguments": "openai/gpt-oss-120b",
                        "opp_arguments": "gemini-2.5-flash",
                        "classification": "gemini-2.5-flash",
                        "result": "cohere"
                    }
                },
                "conversation": {
                    "greeting": conversation_data['greeting'],
                    "history": history_turns,
                    "result": conversation_data['result']
                }
            }

            return render_template(
                'multi_agent_debate.html',
                conversation=conversation_data,
                generation_date=datetime.now()
            )
        except Exception as e:
            logger.error(f"Error running debate: {e}")
            return render_template('multi_agent_debate.html', error=str(e), generation_date=datetime.now())
    # For GET request, render the template without conversation data
    return render_template('multi_agent_debate.html', conversation=None, generation_date=None)

@app.route('/download/debate_log.txt')
def download_txt():
    try:
        if 'last_debate_data' not in globals():
            return jsonify({"error": "No debate data available"}), 404
        debate_data = last_debate_data
        history_text = '\n'.join(debate_data['conversation']['history'])
        content = (
            f"Debate Topic: {debate_data['metadata']['debate_topic']}\n\n"
            f"History:\n{history_text}\n\n"
            f"Result:\n{debate_data['conversation']['result']}"
        )
        return send_file(
            io.BytesIO(content.encode('utf-8')),
            mimetype='text/plain',
            as_attachment=True,
            download_name='debate_log.txt'
        )
    except Exception as e:
        logger.error(f"Error downloading TXT: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/download/debate_log.json')
def download_json():
    try:
        if 'last_debate_data' not in globals():
            return jsonify({"error": "No debate data available"}), 404
        debate_data = last_debate_data
        return send_file(
            io.BytesIO(json.dumps(debate_data, indent=4, ensure_ascii=False).encode('utf-8')),
            mimetype='application/json',
            as_attachment=True,
            download_name='debate_log.json'
        )
    except Exception as e:
        logger.error(f"Error downloading JSON: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)