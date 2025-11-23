from flask import Flask, render_template, request, send_file, jsonify
import json
import os
import re
import requests
from crewai import Agent, Crew, LLM, Task
from dotenv import load_dotenv
from typing import Dict, TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_cohere import ChatCohere
from functools import lru_cache
import logging
from datetime import datetime
import io

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

load_dotenv()

groq_key = os.getenv('groq_api_key')
gemini_key = os.getenv('ARYAN_GEMINI_KEY')
cohere_key = os.getenv('COHERE_API_KEY')
if not all([groq_key, gemini_key, cohere_key]):
    raise ValueError("Missing one or more API keys in environment variables")

gemini_llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=gemini_key,
    temperature=0.7
)
llm_groq = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=groq_key,
    temperature=0.6
)
llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=gemini_key,
    temperature=0.6
)
llm_cohere = ChatCohere(
    api_key=cohere_key,
    temperature=0.6
)

OUTPUT_DIR = os.path.join(os.getcwd(), 'static', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_FILES = 30
def cleanup_old_files():
    files = sorted(
        [os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR)],
        key=os.path.getmtime
    )
    if len(files) > MAX_FILES:
        for old in files[:-MAX_FILES]:
            try:
                os.remove(old)
            except Exception:
                pass

# json cleaner
def extract_json(content: str) -> str:
    """Extract the first JSON object from raw text safely."""
    content = re.sub(r"^```(?:json)?", "", content.strip(), flags=re.IGNORECASE | re.MULTILINE)
    content = re.sub(r"```$", "", content.strip(), flags=re.MULTILINE)
    match = re.search(r'(\{.*\})', content, re.DOTALL)
    return match.group(1).strip() if match else content.strip()

def clean_json_file(file_path):
    try:
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return f"JSON file {file_path} is missing or empty"
        with open(file_path, 'r') as f:
            content = f.read().strip()
        if not content:
            return f"JSON file {file_path} is empty after stripping"
        cleaned_content = extract_json(content)
        try:
            json.loads(cleaned_content)
        except json.JSONDecodeError as e:
            return f"Invalid JSON in {file_path} after cleaning: {str(e)}"
        with open(file_path, 'w') as f:
            f.write(cleaned_content)
        return None
    except Exception as e:
        return str(e)

def validate_mermaid_code(code: str) -> bool:
    """Validate Mermaid code more flexibly."""
    if not code or not isinstance(code, str):
        return False
    if not re.search(r'graph\s+(TD|LR|BT|RL)', code, re.IGNORECASE):
        return False
    if not any(link in code for link in ['-->', '---']):
        return False
    if not re.search(r'\[.*?\]', code) and "subgraph" not in code.lower() and re.search(r'[A-Za-z0-9]+\s*-->', code) is None:
        return False
    return True

def render_with_kroki(mermaid_code, variant, fmt):
    """Render Mermaid code via Kroki API into SVG or PNG."""
    url = f"https://kroki.io/mermaid/{fmt}"
    response = requests.post(url, data=mermaid_code.encode("utf-8"))
    if response.status_code != 200:
        raise ValueError(f"Kroki rendering failed ({fmt}): {response.text}")
    path = os.path.join(OUTPUT_DIR, variant[f"{fmt}_file"])
    mode = "wb" if fmt == "png" else "w"
    with open(path, mode) as f:
        if fmt == "png":
            f.write(response.content)
        else:
            f.write(response.text)
    return f"static/outputs/{variant[f'{fmt}_file']}"

# Flowchart Generator Logic
mermaid_generator = Agent(
    role='Mermaid Code Generator',
    goal='Generate six distinct Mermaid flowchart interpretations with unique logical structures based on a user-provided description.',
    backstory='You are an expert in diagramming, skilled in creating valid and diverse Mermaid syntax for flowcharts tailored for research processes.',
    verbose=True,
    llm=gemini_llm,
    tools=[]
)

flowchart_renderer = Agent(
    role='Flowchart Renderer',
    goal='Render Mermaid flowchart code into SVG and PNG images using the Kroki API.',
    backstory='You are a technical illustrator with expertise in rendering high-quality diagrams for academic and research purposes.',
    verbose=True,
    llm=gemini_llm,
    tools=[]
)

def run_crew(flowchart_description):
    cleanup_old_files()
    variants = [
        {'id': 1, 'name': 'Variant 1', 'json_file': 'mermaid_code_variant1.json', 'svg_file': 'flowchart_output_variant1.svg', 'png_file': 'flowchart_output_variant1.png'},
        {'id': 2, 'name': 'Variant 2', 'json_file': 'mermaid_code_variant2.json', 'svg_file': 'flowchart_output_variant2.svg', 'png_file': 'flowchart_output_variant2.png'},
        {'id': 3, 'name': 'Variant 3', 'json_file': 'mermaid_code_variant3.json', 'svg_file': 'flowchart_output_variant3.svg', 'png_file': 'flowchart_output_variant3.png'},
        {'id': 4, 'name': 'Variant 4', 'json_file': 'mermaid_code_variant4.json', 'svg_file': 'flowchart_output_variant4.svg', 'png_file': 'flowchart_output_variant4.png'},
        {'id': 5, 'name': 'Variant 5', 'json_file': 'mermaid_code_variant5.json', 'svg_file': 'flowchart_output_variant5.svg', 'png_file': 'flowchart_output_variant5.png'},
        {'id': 6, 'name': 'Variant 6', 'json_file': 'mermaid_code_variant6.json', 'svg_file': 'flowchart_output_variant6.svg', 'png_file': 'flowchart_output_variant6.png'}
    ]
    result_json = {'variants': []}
    for variant in variants:
        generate_mermaid_task = Task(
            description=(
                f"Generate a unique Mermaid flowchart in JSON format based on the given research process description.\n\n"
                f"Input Description: {flowchart_description}\n"
                f"Variant: {variant['id']} of 6 — ensure each variant has a distinct logical structure (e.g., linear, branching, parallel, or cyclic) suitable for research workflows.\n\n"
                f"Mermaid Syntax Requirements:\n"
                f"- Start with 'graph TD;'.\n"
                f"- Always wrap every node label in quotes, regardless of content. Example: A[\"Start\"], B{{\"x > y?\"}}, C[\"Return -1 (Not Found)\"].\n"
                f"- Use square brackets for process/statement nodes (e.g., A[\"Do something\"]).\n"
                f"- Use curly braces for decision/condition nodes (e.g., B{{\"Check condition?\"}}).\n"
                f"- Use '-->' for all connections (no extra dashes).\n"
                f"- Never leave raw text or symbols (like /, &, ==, etc..) outside of quotes inside node labels.\n"
                f"- Ensure all brackets and quotes are properly balanced.\n\n"
                f"Output Format:\n"
                f"Return a JSON object containing one key 'mermaid_code' with the Mermaid flowchart string.\n"
                f"Example: {{\"mermaid_code\": \"graph TD; A[Start] --> B[Process Data] --> C[End]\"}}"
            ),
            expected_output="A JSON object with valid Mermaid flowchart code under 'mermaid_code'.",
            agent=mermaid_generator,
            output_file=os.path.join(OUTPUT_DIR, variant['json_file'])
        )
        crew = Crew(
            agents=[mermaid_generator],
            tasks=[generate_mermaid_task],
            verbose=True
        )
        try:
            crew.kickoff()
            json_file = os.path.join(OUTPUT_DIR, variant['json_file'])
            if not os.path.exists(json_file):
                result_json['variants'].append({'id': variant['id'], 'name': variant['name'], 'error': f"JSON file missing: {json_file}"})
                continue
            json_error = clean_json_file(json_file)
            if json_error:
                result_json['variants'].append({'id': variant['id'], 'name': variant['name'], 'error': f"Failed to clean JSON file: {json_error}"})
                continue
            with open(json_file, 'r') as f:
                mermaid_data = json.load(f)
            mermaid_code = mermaid_data.get('mermaid_code', '')
            if not validate_mermaid_code(mermaid_code):
                result_json['variants'].append({'id': variant['id'], 'name': variant['name'], 'error': 'Invalid or empty Mermaid code'})
                continue
            variant_result = {'id': variant['id'], 'name': variant['name']}
            try:
                variant_result['svg_path'] = render_with_kroki(mermaid_code, variant, "svg")
                variant_result['png_path'] = render_with_kroki(mermaid_code, variant, "png")
                result_file = os.path.join(OUTPUT_DIR, f"flowchart_result_variant{variant['id']}.json")
                with open(result_file, 'w') as f:
                    json.dump(variant_result, f, indent=2)
                result_json['variants'].append(variant_result)
            except Exception as e:
                variant_result['error'] = f"Rendering failed: {str(e)}"
                result_json['variants'].append(variant_result)
        except Exception as e:
            result_json['variants'].append({'id': variant['id'], 'name': variant['name'], 'error': f"Crew execution error: {str(e)}"})
    return result_json

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
        output = sanitize_text(llm_groq.invoke("I wish to have a debate on {}. What would be the fighting sides called? Output just the names and nothing else as comma separated list".format(debate_topic)).content)
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
            result = sanitize_text(llm_gemini.invoke("classify the sentiment of input as {} or {}. Output just the class. Input:{}".format(class0, class1, question)).content.strip())
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
            argument = classes[0] + ": " + sanitize_text(llm_groq.invoke(prompt).content)
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
            argument = classes[1] + ": " + sanitize_text(llm_gemini.invoke(prompt).content)
            logger.info(f"Opp argument: {argument}")
            return {"history": summary + '\n' + argument, "current_response": argument, "count": state.get('count', 0) + 1}
        except Exception as e:
            logger.error(f"Error in handle_opp: {e}")
            return {"history": summary + '\n' + f"{classes[1]}: Error generating argument", "current_response": "Error", "count": state.get('count', 0) + 1}

    def result(state):
        try:
            summary = state.get('history', '').strip()
            prompt = "Summarize the conversation and judge who won the debate. No ties are allowed. Conversation: {}".format(summary)
            result = sanitize_text(llm_cohere.invoke(prompt).content)
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

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/flowchart_generator', methods=['GET', 'POST'])
def flowchart_generator():
    result = None
    error = None
    if request.method == 'POST':
        flowchart_description = request.form.get('description')
        if not flowchart_description:
            error = "Please provide a flowchart description."
        else:
            result = run_crew(flowchart_description)
            if not result.get('variants'):
                error = "No flowchart variants were generated."
            elif all('error' in v for v in result['variants']):
                error = "All variants failed: " + "; ".join(v['error'] for v in result['variants'])
                result = None
            elif any('error' in v for v in result['variants']):
                error = "Some variants failed. See details below."
    return render_template('flowchart_generator.html', result=result or {}, error=error)

@app.route('/multi-agent-debate', methods=['GET', 'POST'])
def multi_agent_debate():
    if request.method == 'POST':
        try:
            debate_topic = request.form.get('debate_topic', '').strip()
            max_turns = int(request.form.get('max_turns', 10))
            if not debate_topic:
                return render_template('multi_agent_debate.html', error="Debate topic is required", generation_date=datetime.now())
            workflow, classes = create_debate_workflow(debate_topic=debate_topic, max_turns=max_turns)
            app_workflow = workflow.compile()
            conversation = app_workflow.invoke({'count': 0, 'history': '', 'current_response': ''})
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
            conversation_data = {
                'greeting': conversation.get('greeting', 'No greeting generated'),
                'history': history_turns,
                'result': conversation.get('results', 'No result generated')
            }
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
    return render_template('multi_agent_debate.html', conversation=None, generation_date=None)

@app.route('/download/<file_type>/<variant_id>')
def download(file_type, variant_id):
    variant_files = {
        '1': {'svg': 'flowchart_output_variant1.svg', 'png': 'flowchart_output_variant1.png'},
        '2': {'svg': 'flowchart_output_variant2.svg', 'png': 'flowchart_output_variant2.png'},
        '3': {'svg': 'flowchart_output_variant3.svg', 'png': 'flowchart_output_variant3.png'},
        '4': {'svg': 'flowchart_output_variant4.svg', 'png': 'flowchart_output_variant4.png'},
        '5': {'svg': 'flowchart_output_variant5.svg', 'png': 'flowchart_output_variant5.png'},
        '6': {'svg': 'flowchart_output_variant6.svg', 'png': 'flowchart_output_variant6.png'}
    }
    if variant_id not in variant_files or file_type not in ['svg', 'png']:
        return jsonify({'error': 'Invalid file type or variant ID'}), 400
    file_path = os.path.join(OUTPUT_DIR, variant_files[variant_id][file_type])
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return jsonify({'error': f'{file_type.upper()} file missing or empty for variant {variant_id}'}), 404
    return send_file(file_path, as_attachment=True, download_name=variant_files[variant_id][file_type])

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

if __name__ == "__main__":
    app.run(debug=True)