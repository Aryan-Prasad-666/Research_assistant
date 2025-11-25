from flask import Flask, render_template, request, send_file, jsonify
import json
import os
import re
import requests
from crewai import Agent, Crew, LLM, Task
from crewai_tools import SerperDevTool, ArxivPaperTool
from dotenv import load_dotenv
from typing import Dict, TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import ChatCohere
from langchain.prompts import PromptTemplate
import logging
from datetime import datetime
import io
from openai import OpenAI
from supermemory import Supermemory
import PyPDF2 
import uuid
from werkzeug.utils import secure_filename 
from langchain_core.output_parsers import JsonOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from PyPDF2 import PdfReader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

load_dotenv()

groq_key = os.getenv('groq_api_key')
gemini_key = os.getenv('ARYAN_GEMINI_KEY')
cohere_key = os.getenv('COHERE_API_KEY')
supermemory_key = os.getenv('ARYAN_SUPERMEMORY_API_KEY')
serper_key = os.getenv('SERPER_API_KEY')

if not all([groq_key, gemini_key, cohere_key, supermemory_key]):
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

serper_tool = SerperDevTool(api_key=serper_key, n_results=5) 
arxiv_tool = ArxivPaperTool()

client = Supermemory(api_key=supermemory_key)
groq_client = OpenAI(
    api_key=groq_key,
    base_url="https://api.supermemory.ai/v3/https://api.groq.com/openai/v1",
    default_headers={
        "x-supermemory-api-key": supermemory_key,
        "x-sm-user-id": "user_123" 
    }
)

research_prompt = PromptTemplate(
    input_variables=["user_query"],
    template="""You are an expert research assistant powered by advanced AI, specialized exclusively in academic and scientific research support. Your goal is to provide accurate, insightful, and well-structured responses ONLY to research-oriented queries, such as literature reviews, methodology design, data analysis, hypothesis formulation, citation management, gap identification, experimental design, or scholarly writing.

If the user's query is unrelated to research (e.g., personal finance, general advice, or non-academic topics), politely decline to answer and redirect them to research-related questions. For example: "I'm here to assist with research tasks. Could you tell me more about your study or project?"

Draw from our shared conversation history and knowledge to personalize and enhance your assistance, but always stay within the bounds of research support.

User Query: {user_query}

Respond only with pure text. Do not use any special symbols, markdown, or formatting like *, |, #, etc. Keep responses clear and professional.
"""
)

OUTPUT_DIR = os.path.join(os.getcwd(), 'static', 'outputs') 
GAP_OUTPUT_DIR = os.path.join(os.getcwd(), 'static', 'gaps') 
UPLOAD_DIR = os.path.join(os.getcwd(), 'Uploads') 
SUMMARY_OUTPUT_DIR = os.path.join(os.getcwd(), 'static', 'summaries')
VECTOR_DB_DIR = os.path.join(os.getcwd(), 'vector_db')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GAP_OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SUMMARY_OUTPUT_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

MAX_FILES = 30
def cleanup_old_files(directory):
    """Clean up old files in a specific directory."""
    files = sorted(
        [os.path.join(directory, f) for f in os.listdir(directory)],
        key=os.path.getmtime
    )
    if len(files) > MAX_FILES:
        for old in files[:-MAX_FILES]:
            try:
                os.remove(old)
                logger.info(f"Removed old file: {old} from {directory}")
            except Exception as e:
                logger.warning(f"Failed to remove old file {old} from {directory}: {e}")

def run_cleanup():
    cleanup_old_files(OUTPUT_DIR)
    cleanup_old_files(GAP_OUTPUT_DIR)
    cleanup_old_files(SUMMARY_OUTPUT_DIR)

def extract_json(content: str) -> str:
    """Extract the first JSON object from raw text safely."""
    content = re.sub(r"^```(?:json)?", "", content.strip(), flags=re.IGNORECASE | re.MULTILINE)
    content = re.sub(r"```$", "", content.strip(), flags=re.MULTILINE)
    match = re.search(r'(\{.*\})', content, re.DOTALL)
    return match.group(1).strip() if match else content.strip()

def clean_json_file(file_path):
    """Clean and validate JSON file content."""
    try:
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return f"JSON file {file_path} is missing or empty"
        with open(file_path, 'r', encoding='utf-8') as f: 
            content = f.read().strip()
        if not content:
            return f"JSON file {file_path} is empty after stripping"
        cleaned_content = extract_json(content)
        try:
            json.loads(cleaned_content)
        except json.JSONDecodeError as e:
            return f"Invalid JSON in {file_path} after cleaning: {str(e)}"
        with open(file_path, 'w', encoding='utf-8') as f: 
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

def extract_text_from_pdf(file_input) -> Optional[str]:
    """
    Extract text from a PDF file stream (used for Summarizer) or a file path (used for Gap Finder).
    File_input can be a file stream from request.files or a string file path.
    """
    try:
        if isinstance(file_input, str):
            with open(file_input, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"
            logger.info(f"Successfully extracted text from PDF path: {file_input}")
        else:
            pdf_reader = PyPDF2.PdfReader(file_input)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
            logger.info("Successfully extracted text from PDF stream.")

        return text.strip() if text.strip() else None
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return None

def sanitize_text(text: str) -> str:
    """Replace or remove problematic Unicode characters."""
    if not isinstance(text, str):
        return text
    text = text.replace('\u2011', '-')
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text

gap_analyzer_agent = Agent(
    role='Research Gap Analyzer',
    goal='Analyze uploaded literature review text to identify key research gaps, under-explored areas, and opportunities for future work. Output directly as structured JSON.',
    backstory='You are an expert academic analyst specializing in literature reviews. You identify gaps by examining themes, methodologies, findings, and limitations in the provided text.',
    verbose=True,
    llm=gemini_llm,
    tools=[]
)

def run_gap_finder_crew(text: str, document_id: str) -> Dict:
    """Run the Gap Finder crew with a single main agent to analyze and format gaps."""
    cleanup_old_files(GAP_OUTPUT_DIR)

    analyze_gaps_task = Task(
        description=(
            f"Analyze the following literature review text to identify 5-8 major research gaps. "
            f"Focus on under-explored areas, methodological limitations, unanswered questions, and future directions. "
            f"Text: {text[:4000]}...\n\n"
            f"Output ONLY a valid JSON object with a 'gaps' key containing an array of objects. "
            f"Each gap object must have: 'title' (concise 1-sentence summary of the gap) and 'description' (detailed explanation with evidence from text, 2-4 sentences). "
            f"Example: {{\"gaps\": [{{\"title\": \"Gap in X\", \"description\": \"Detailed explanation...\"}}]}}"
        ),
        expected_output="JSON object with 'gaps' array of objects containing 'title' and 'description' keys.",
        agent=gap_analyzer_agent,
        output_file=os.path.join(GAP_OUTPUT_DIR, f'gaps_{document_id}.json')
    )

    try:
        crew = Crew(
            agents=[gap_analyzer_agent],
            tasks=[analyze_gaps_task],
            verbose=True
        )
        crew.kickoff()

        output_file = os.path.join(GAP_OUTPUT_DIR, f'gaps_{document_id}.json')
        json_error = clean_json_file(output_file)
        if json_error:
            logger.error(f"JSON cleaning error for {output_file}: {json_error}")
            return {"error": f"Failed to process gaps: {json_error}"}

        with open(output_file, 'r', encoding='utf-8') as f:
            result_data = json.load(f)

        gaps = result_data.get('gaps', [])
        if not isinstance(gaps, list) or not gaps:
            return {"error": "No valid gaps identified in analysis."}

        logger.info(f"Successfully identified {len(gaps)} gaps for document {document_id}")
        return {"gaps": gaps}
    except Exception as e:
        logger.error(f"Error running gap finder crew: {e}")
        return {"error": f"Analysis failed: {str(e)}"}

def create_vector_store(text: str, document_id: str) -> Chroma:
    """Create a vector store from text chunks."""
    chunks = text_splitter.split_text(text)
    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name=document_id,
        persist_directory=VECTOR_DB_DIR
    )
    vector_store.persist()
    logger.info(f"Created vector store for document_id: {document_id}")
    return vector_store

def generate_summary(vector_store: Chroma, document_id: str) -> Dict:
    """Generate a structured summary using retrieved chunks."""
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        chunks = retriever.get_relevant_documents("summarize the key points, methodologies, and findings")
        
        combined_text = "\n".join([chunk.page_content for chunk in chunks])
        prompt = (
            f"Summarize the following document content in a structured JSON format. "
            f"Include sections for 'key_points', 'methodologies', and 'findings'. "
            f"Each section should contain a concise list of bullet points (max 5 per section). "
            f"Ensure clarity and avoid repetition.\n\n"
            f"Document Content:\n{combined_text}\n\n"
            f"Output Format:\n"
            f"{{\"key_points\": [], \"methodologies\": [], \"findings\": []}}"
        )
        parser = JsonOutputParser()
        summary = parser.parse(sanitize_text(llm_gemini.invoke(prompt).content))
        logger.info(f"Generated summary for document_id: {document_id}")
        return summary
    except Exception as e:
        logger.error(f"Error generating summary for document_id {document_id}: {e}")
        return {"error": str(e)}

def detect_bias(vector_store: Chroma, document_id: str) -> Dict:
    """Detect potential biases in tone, framing, and source diversity."""
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        chunks = retriever.get_relevant_documents("analyze tone, framing, and sources")
        
        combined_text = "\n".join([chunk.page_content for chunk in chunks])
        prompt = (
            f"Analyze the following document content for potential biases in tone, framing, and source diversity. "
            f"Return a JSON object with sections for 'tone', 'framing', 'source_diversity'. "
            f"Each section should include a 'description' (brief analysis) and 'flags' (list of specific issues, max 3). "
            f"Output Format:\n"
            f"{{\"tone\": {{\"description\": \"\", \"flags\": []}}, "
            f"\"framing\": {{\"description\": \"\", \"flags\": []}}, "
            f"\"source_diversity\": {{\"description\": \"\", \"flags\": []}}}}"
            f"\n\nDocument Content:\n{combined_text}"
        )
        parser = JsonOutputParser()
        bias_analysis = parser.parse(sanitize_text(llm_gemini.invoke(prompt).content))
        logger.info(f"Generated bias analysis for document_id: {document_id}")
        return bias_analysis
    except Exception as e:
        logger.error(f"Error detecting bias for document_id {document_id}: {e}")
        return {"error": str(e)}

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
    cleanup_old_files(OUTPUT_DIR) 
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
                        f"STRICT OUTPUT REQUIREMENT:\n"
                        f"You MUST output ONLY the raw JSON object. Do not include any introductory text, markdown fences (```json or ```), or explanations before or after the JSON object.\n"
                        f"The value for 'mermaid_code' MUST be a single, valid Mermaid string.\n\n"
                        f"Mermaid Syntax Requirements for VALID CODE:\n"
                        f"1. Start with 'graph TD;' or 'graph LR;'.\n"
                        f"2. All node labels MUST be enclosed in quotes (e.g., A[\"Start Process\"], B{{\"Decision Point?\"}}).\n"
                        f"3. Conditional links must be correctly labeled: e.g., B -- Yes --> C;\n"
                        f"4. Node IDs must be alphanumeric and simple (e.g., N1, P2).\n"
                        f"5. Ensure all quotes and parentheses are balanced.\n\n"
                        f"Output Format (STRICTLY ADHERE TO THIS):\n"
                        f"{{\"mermaid_code\": \"graph TD; A[\"Start Research\"] --> B{{\"Review Complete?\"}}; B -- Yes --> C[\"Define Methodology\"]; B -- No --> A; C --> D[\"End\"]\"}}"
                    ),
                    expected_output="A raw JSON object with valid Mermaid flowchart code under 'mermaid_code'.",
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
                result_json['variants'].append({'id': variant['id'], 'name': variant['name'], 'error': 'Invalid Mermaid code generated'})
                continue
            for fmt in ['svg', 'png']:
                try:
                    render_with_kroki(mermaid_code, variant, fmt)
                except Exception as render_err:
                    result_json['variants'].append({'id': variant['id'], 'name': variant['name'], 'error': f'Render failed for {fmt}: {str(render_err)}'})
                    break
            result_json['variants'].append({'id': variant['id'], 'name': variant['name']})
        except Exception as e:
            logger.error(f"Error generating variant {variant['id']}: {e}")
            result_json['variants'].append({'id': variant['id'], 'name': variant['name'], 'error': str(e)})
    return result_json

def sanitize_text_debate(text): 
    return re.sub(r'\*\*(.*?)\*\*', r'\1', text)

def create_debate_workflow(debate_topic, max_turns):
    classes = ["Proponent", "Opponent"]
    pro_llm = llm_groq
    opp_llm = llm_gemini
    classify_llm = llm_gemini

    class DebateState(TypedDict):
        history: str
        classification: str
        greeting: str
        current_response: str
        count: int
        results: str

    workflow = StateGraph(DebateState)

    def handle_greeting_node(state):
        prompt = f"Create a neutral introduction to the debate on: '{debate_topic}'. Keep it engaging and informative."
        greeting = sanitize_text_debate(llm_cohere.invoke(prompt).content)
        logger.info(f"Generated greeting: {greeting}")
        return {"greeting": greeting, "history": f"Introduction: {greeting}", "count": 0}

    def classify_input_node(state):
        history = state.get('history', '')
        prompt = f"Classify the next response in this debate history as either 'proponent' or 'opponent' based on who should speak next. Debate topic: '{debate_topic}'. History: {history}. Respond with only one word: 'proponent' or 'opponent'."
        classification = sanitize_text_debate(classify_llm.invoke(prompt).content).strip().lower()
        if classification not in ['proponent', 'opponent']:
            classification = 'proponent'  
        logger.info(f"Classified next speaker: {classification}")
        return {"classification": classification}

    def handle_pro(state):
        history = state.get('history', '')
        prompt = f"As the Proponent, provide a strong, evidence-based argument supporting the topic: '{debate_topic}'. Build on previous arguments. Keep concise (2-4 paragraphs). History: {history}"
        response = sanitize_text_debate(pro_llm.invoke(prompt).content)
        logger.info(f"Proponent response: {response}")
        return {"history": history + '\n' + f"{classes[0]}: {response}", "current_response": response, "count": state.get('count', 0) + 1}

    def handle_opp(state):
        history = state.get('history', '')
        prompt = f"As the Opponent, provide a compelling counter-argument against the topic: '{debate_topic}'. Use logic and evidence. Keep concise (2-4 paragraphs). History: {history}"
        response = sanitize_text_debate(opp_llm.invoke(prompt).content)
        logger.info(f"Opponent response: {response}")
        return {"history": history + '\n' + f"{classes[1]}: {response}", "current_response": response, "count": state.get('count', 0) + 1}

    def result(state):
        try:
            summary = state.get('history', '').strip()
            prompt = "Summarize the conversation and judge who won the debate. No ties are allowed. Conversation: {}".format(summary)
            result = sanitize_text_debate(llm_cohere.invoke(prompt).content)
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

youtube_agent = Agent(
    role='YouTube Resource Fetcher',
    goal='Fetch relevant YouTube videos, tutorials, lectures, and explainers based on the query using web search.',
    backstory='You are an expert in sourcing educational video content from YouTube using web search tools for research and learning purposes.',
    verbose=True,
    llm=gemini_llm,
    tools=[serper_tool]
)

serper_agent = Agent(
    role='Web Resource Fetcher',
    goal='Search the web for blogs, online courses (MOOCs), tutorials, documentation, and Q&A forums using Google search, returning the top 5 most relevant results.',
    backstory='You are a web researcher skilled in aggregating diverse online resources from trusted sources.',
    verbose=True,
    llm=gemini_llm,
    tools=[serper_tool]
)

arxiv_agent = Agent(
    role='arXiv Resource Fetcher',
    goal='Pull recent or highly cited research papers, abstracts, categories, and download links from arXiv.',
    backstory='You are an academic paper scout focused on scientific and technical publications.',
    verbose=True,
    llm=gemini_llm,
    tools=[arxiv_tool]
)

aggregator_agent = Agent(
    role='Resource Aggregator',
    goal='Merge and organize results from YouTube, Serper, and arXiv tasks into categorized sections, formatted as HTML lists for frontend display.',
    backstory='You are a knowledge integrator that combines multi-source data into a cohesive, display-ready structure.',
    verbose=True,
    llm=gemini_llm,
    tools=[]
)

def run_knowledge_crew(query):
    """Run the Knowledge Hub crew to fetch and aggregate resources for the given query."""
    cleanup_old_files(OUTPUT_DIR) 

    youtube_task = Task(
        description=f"Use Google search with 'site:youtube.com' to find tutorials, lectures, and explainers on '{query}'. Fetch top 5 results with titles, links, and brief descriptions. Return results as a JSON array of objects with keys: title, link, description.",
        expected_output='JSON array of YouTube resources, e.g., [{"title": "Video Title", "link": "url", "description": "desc"}]',
        agent=youtube_agent,
        output_file=os.path.join(OUTPUT_DIR, 'youtube_results.json'),
        callback=lambda x: logger.info(f"YouTube task output: {x}")
    )

    serper_task = Task(
        description=f"Use Google search to find blogs, MOOCs, tutorials, docs, and forums on '{query}'. Fetch top 5 relevant results with titles, snippets, and links. Return results as a JSON array of objects with keys: title, snippet, link.",
        expected_output='JSON array of web resources, e.g., [{"title": "Blog Title", "snippet": "snippet", "link": "url"}]',
        agent=serper_agent,
        output_file=os.path.join(OUTPUT_DIR, 'serper_results.json'),
        callback=lambda x: logger.info(f"Serper task output: {x}")
    )

    arxiv_task = Task(
        description=f"Query arXiv for papers on '{query}'. Fetch top 5 recent or highly cited papers with titles, authors, abstracts, categories, and PDF links. Return results as a JSON array of objects with keys: title, authors, abstract, categories, pdf_link.",
        expected_output='JSON array of arXiv papers, e.g., [{"title": "Paper Title", "authors": "Author List", "abstract": "abstract", "categories": "category", "pdf_link": "url"}]',
        agent=arxiv_agent,
        output_file=os.path.join(OUTPUT_DIR, 'arxiv_results.json'),
        callback=lambda x: logger.info(f"arXiv task output: {x}")
    )

    def clean_task_output(file_path):
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f: 
                    raw_content = f.read()
                    logger.info(f"Raw content of {file_path}: {raw_content[:200]}...")
                    cleaned_content = extract_json(raw_content)
                    json.loads(cleaned_content)
                    with open(file_path, 'w', encoding='utf-8') as f: 
                        f.write(cleaned_content)
                    logger.info(f"Cleaned JSON for {file_path}: {cleaned_content[:200]}...")
            else:
                logger.warning(f"File not found: {file_path}")
                with open(file_path, 'w', encoding='utf-8') as f: 
                    f.write('[]') 
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in {file_path}: {str(e)}")
            with open(file_path, 'w', encoding='utf-8') as f: 
                f.write('[]') 
        except Exception as e:
            logger.error(f"Error cleaning {file_path}: {str(e)}")
            with open(file_path, 'w', encoding='utf-8') as f: 
                f.write('[]') 

    crew = Crew(
        agents=[youtube_agent, serper_agent, arxiv_agent],
        tasks=[youtube_task, serper_task, arxiv_task],
        verbose=True
    )
    crew.kickoff()

    clean_task_output(os.path.join(OUTPUT_DIR, 'youtube_results.json'))
    clean_task_output(os.path.join(OUTPUT_DIR, 'serper_results.json'))
    clean_task_output(os.path.join(OUTPUT_DIR, 'arxiv_results.json'))

    aggregate_task = Task(
        description=(
            f"Aggregate results from YouTube, Serper, and arXiv tasks into a single structured JSON with categories: Tutorials (YouTube), Blogs & MOOCs (Serper), Research (arXiv). "
            f"For each category, format the results as an HTML unordered list (<ul>) with each item as an <li> containing relevant details (e.g., title, link, description or summary). "
            f"Ensure links are clickable (<a href>). If a category has no results, include an empty list (<ul></ul>). "
            f"Example: "
            f'{{"Tutorials": "<ul><li><a href=\'url\'>Video Title</a>: desc</li></ul>", '
            f'"Research": "<ul><li><a href=\'url\'>Paper Title</a>: abstract</li></ul>", '
            f'"Blogs & MOOCs": "<ul><li><a href=\'url\'>Blog Title</a>: snippet</li></ul>"}}'
        ),
        expected_output="JSON object with categorized resources formatted as HTML lists.",
        agent=aggregator_agent,
        context=[youtube_task, serper_task, arxiv_task],
        output_file=os.path.join(OUTPUT_DIR, 'aggregated_results.json'),
        callback=lambda x: logger.info(f"Aggregate task output: {x}")
    )

    try:
        crew = Crew(
            agents=[aggregator_agent],
            tasks=[aggregate_task],
            verbose=True
        )
        crew.kickoff()

        aggregated_file = os.path.join(OUTPUT_DIR, 'aggregated_results.json')
        if os.path.exists(aggregated_file):
            with open(aggregated_file, 'r', encoding='utf-8') as f: 
                raw_content = f.read()
                logger.info(f"Raw content of aggregated_results.json: {raw_content[:200]}...")
                cleaned_content = extract_json(raw_content)
                logger.info(f"Cleaned content of aggregated_results.json: {cleaned_content[:200]}...")
                result = json.loads(cleaned_content)
            logger.info(f"Aggregated results: {result}")
            return result
        else:
            logger.error(f"Aggregated results file not found: {aggregated_file}")
            return {"error": "Aggregated results file not found"}
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error in aggregated_results.json: {str(e)}")
        return {"error": f"Failed to parse aggregated results: {str(e)}"}
    except Exception as e:
        logger.error(f"Error running knowledge crew: {e}")
        return {"error": f"Failed to fetch resources: {str(e)}"}

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/chatbot')
def chatbot():
    current_time = datetime.now().strftime('%H:%M')
    return render_template('chat.html', current_time=current_time)

@app.route('/chatbot/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message')
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        system_prompt = research_prompt.format(
            user_query=user_message
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )

        ai_response = response.choices[0].message.content

        return jsonify({'response': ai_response})

    except Exception as error:
        return jsonify({'error': str(error)}), 500

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

@app.route('/gap_finder', methods=['GET', 'POST'])
def gap_finder():
    result = None
    error = None
    if request.method == 'POST':
        if 'file' not in request.files:
            error = "No file uploaded."
            logger.warning("No file in request")
        else:
            file = request.files['file']
            if file.filename == '':
                error = "No file selected."
                logger.warning("Empty filename")
            elif not file.filename.lower().endswith('.pdf'):
                error = "Only PDF files are supported."
                logger.warning(f"Invalid file type uploaded: {file.filename}")
            else:
                document_id = str(uuid.uuid4())
                filename = secure_filename(file.filename)
                file_path = os.path.join(UPLOAD_DIR, f"{document_id}_{filename}")
                file.save(file_path)
                logger.info(f"File saved to {file_path}")
                
                run_cleanup() 

                text = extract_text_from_pdf(file_path)
                if not text:
                    error = "Failed to extract text from PDF."
                    logger.error(f"Failed to extract text from PDF: {file_path}")
                else:
                    gaps = run_gap_finder_crew(text, document_id)
                    if "error" in gaps:
                        error = f"Error processing document: {gaps['error']}"
                        logger.error(f"Processing error for document_id {document_id}: {error}")
                    else:
                        result = {
                            "gaps": gaps["gaps"],
                            "document_id": document_id
                        }
                        output_file = os.path.join(GAP_OUTPUT_DIR, f"gaps_{document_id}.json")
                        try:
                            with open(output_file, 'w', encoding='utf-8') as f:
                                json.dump(result, f, indent=4, ensure_ascii=False)
                            logger.info(f"Saved gap analysis to {output_file}")
                        except Exception as e:
                            error = f"Failed to save gap analysis file: {str(e)}"
                            logger.error(f"Error saving gap analysis to {output_file}: {e}")
    
    logger.info(f"Rendering gap_finder.html with result: {result}, error: {error}")
    return render_template('gap_finder.html', result=result or {}, error=error, generation_date=datetime.now())

@app.route('/summarizer', methods=['GET', 'POST'])
def summarize():
    result = None
    error = None
    document_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if request.method == 'POST':
        if 'file' not in request.files or not request.files['file'].filename:
            error = "Please upload a PDF file."
            logger.warning("No file uploaded in POST request")
        else:
            file = request.files['file']
            if not file.filename.lower().endswith('.pdf'):
                error = "Only PDF files are supported."
                logger.warning(f"Invalid file type uploaded: {file.filename}")
            else:
                run_cleanup() 

                text = extract_text_from_pdf(file)
                if not text:
                    error = "Failed to extract text from PDF."
                    logger.error("Failed to extract text from uploaded PDF")
                else:
                    vector_store = create_vector_store(text, document_id)
                    
                    summary = generate_summary(vector_store, document_id)
                    bias_analysis = detect_bias(vector_store, document_id)
                    
                    if "error" in summary or "error" in bias_analysis:
                        error = "Error processing document: " + (summary.get("error", "") or bias_analysis.get("error", ""))
                        logger.error(f"Processing error for document_id {document_id}: {error}")
                    else:
                        result = {
                            "summary": summary,
                            "bias_analysis": bias_analysis,
                            "document_id": document_id
                        }
                        output_file = os.path.join(SUMMARY_OUTPUT_DIR, f"summary_{document_id}.json")
                        try:
                            with open(output_file, 'w', encoding='utf-8') as f:
                                json.dump(result, f, indent=4, ensure_ascii=False)
                            logger.info(f"Saved summary to {output_file}")
                        except Exception as e:
                            error = f"Failed to save summary file: {str(e)}"
                            logger.error(f"Error saving summary to {output_file}: {e}")
    
    return render_template('summarize.html', result=result or {}, error=error, generation_date=datetime.now())

@app.route('/download/gaps/<document_id>.json')
def download_gaps(document_id):
    try:
        file_path = os.path.join(GAP_OUTPUT_DIR, f"gaps_{document_id}.json")
        logger.info(f"Attempting to download file: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return jsonify({"error": f"Gap analysis file for document ID {document_id} not found"}), 404
        
        if not os.access(file_path, os.R_OK):
            logger.error(f"No read permission for file: {file_path}")
            return jsonify({"error": f"No read permission for gap analysis file {document_id}"}), 403
        
        file_size = os.path.getsize(file_path)
        logger.info(f"File size for {file_path}: {file_size} bytes")
        
        return send_file(
            file_path,
            mimetype='application/json',
            as_attachment=True,
            download_name=f"gaps_{document_id}.json"
        )
    except Exception as e:
        logger.error(f"Error downloading gap analysis for document_id {document_id}: {e}")
        return jsonify({"error": f"Failed to download gap analysis: {str(e)}"}), 500

@app.route('/download/summary/<document_id>.json')
def download_summary(document_id):
    try:
        file_path = os.path.join(SUMMARY_OUTPUT_DIR, f"summary_{document_id}.json")
        logger.info(f"Attempting to download file: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return jsonify({"error": f"Summary file for document ID {document_id} not found"}), 404
        
        if not os.access(file_path, os.R_OK):
            logger.error(f"No read permission for file: {file_path}")
            return jsonify({"error": f"No read permission for summary file {document_id}"}), 403
        
        file_size = os.path.getsize(file_path)
        logger.info(f"File size for {file_path}: {file_size} bytes")
        
        return send_file(
            file_path,
            mimetype='application/json',
            as_attachment=True,
            download_name=f"summary_{document_id}.json"
        )
    except Exception as e:
        logger.error(f"Error downloading summary for document_id {document_id}: {str(e)}")
        return jsonify({"error": f"Failed to download summary: {str(e)}"}), 500

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

@app.route('/knowledge_hub', methods=['GET', 'POST'])
def knowledge_hub():
    """Handle Knowledge Hub requests and render the results."""
    result = None
    error = None
    query = None
    if request.method == 'POST':
        query = request.form.get('query')
        if not query:
            error = "Please provide a search query."
        else:
            result = run_knowledge_crew(query)
            logger.info(f"Result returned to template: {result}")
            if 'error' in result:
                error = result['error']
                result = None
    return render_template('knowledge_hub.html', result=result, error=error, query=query, generation_date=datetime.now(), debug=app.debug)

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