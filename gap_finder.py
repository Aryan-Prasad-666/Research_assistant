import os
import json
import logging
from datetime import datetime
from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from crewai import Agent, Crew, Task, LLM
import PyPDF2
import re
from typing import Dict, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()

# Validate environment variables
gemini_key = os.getenv('gemini_api_key') or os.getenv('ARYAN_API_KEY')
if not gemini_key:
    raise ValueError("Missing Gemini API key in environment variables")

# Initialize LLM using crew.LLM
llm_gemini = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=gemini_key,
    temperature=0.7
)

# Directory for output files
OUTPUT_DIR = os.path.join(os.getcwd(), 'static', 'gaps')
UPLOAD_DIR = os.path.join(os.getcwd(), 'Uploads')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Keep only the last N files to avoid filling disk
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
                logger.info(f"Removed old file: {old}")
            except Exception as e:
                logger.warning(f"Failed to remove old file {old}: {e}")

# Utility Functions
def extract_text_from_pdf(file_path: str) -> Optional[str]:
    """Extract text from a PDF file."""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        logger.info(f"Successfully extracted text from PDF: {file_path}")
        return text.strip() if text.strip() else None
    except Exception as e:
        logger.error(f"Error extracting text from PDF {file_path}: {e}")
        return None

def sanitize_text(text: str) -> str:
    """Replace or remove problematic Unicode characters."""
    if not isinstance(text, str):
        return text
    text = text.replace('\u2011', '-')
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text

def extract_json(content: str) -> str:
    """Extract the first JSON object from raw text safely."""
    content = re.sub(r"^```(?:json)?", "", content.strip(), flags=re.IGNORECASE | re.MULTILINE)
    content = re.sub(r"```$", "", content.strip(), flags=re.MULTILINE)
    match = re.search(r'(\{.*\})', content, re.DOTALL)
    return match.group(1).strip() if match else content.strip()

def clean_json_file(file_path: str) -> Optional[str]:
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

# CrewAI Agents
gap_analyzer = Agent(
    role='Gap Analyzer',
    goal='Analyze a research paper or report to identify knowledge gaps in methodologies, research questions, and source diversity.',
    backstory='You are an expert academic reviewer with a keen eye for identifying missing or unclear elements in research documents.',
    verbose=True,
    llm=llm_gemini,
    tools=[]
)

gap_formatter = Agent(
    role='Gap Formatter',
    goal='Format gap analysis results into structured JSON with clear descriptions and suggestions.',
    backstory='You are a data formatting specialist skilled in presenting academic analysis in a clear, structured JSON format.',
    verbose=True,
    llm=llm_gemini,
    tools=[]
)

def run_gap_finder_crew(text: str, document_id: str) -> Dict:
    """Run CrewAI to generate gap analysis for the provided text."""
    cleanup_old_files()

    # Task 1: Analyze gaps
    analyze_gaps_task = Task(
        description=(
            f"Analyze the following document content to identify knowledge gaps in a research paper or report. "
            f"Focus on three areas:\n"
            f"- Methodological Gaps: Missing or unclear experimental details, methodologies, or data analysis techniques.\n"
            f"- Unaddressed Research Questions: Questions or hypotheses not explored or answered.\n"
            f"- Source Gaps: Lack of diverse citations or reliance on limited sources.\n"
            f"For each gap, provide a brief description (1-2 sentences) and a suggestion for improvement (1 sentence).\n"
            f"Document Content (first 10000 characters):\n{sanitize_text(text[:10000])}\n\n"
            f"Output Format: Return a JSON object with the structure:\n"
            f"{{\"methodological_gaps\": [{{\"description\": \"\", \"suggestion\": \"\"}}], "
            f"\"unaddressed_questions\": [{{\"description\": \"\", \"suggestion\": \"\"}}], "
            f"\"source_gaps\": [{{\"description\": \"\", \"suggestion\": \"\"}}]}}"
        ),
        expected_output="A JSON object containing gap analysis with methodological_gaps, unaddressed_questions, and source_gaps.",
        agent=gap_analyzer,
        output_file=os.path.join(OUTPUT_DIR, f"raw_gaps_{document_id}.json")
    )

    # Task 2: Format gaps
    format_gaps_task = Task(
        description=(
            f"Take the raw gap analysis from the previous task and format it into a clean JSON structure. "
            f"Ensure descriptions are concise (1-2 sentences) and suggestions are actionable (1 sentence). "
            f"Remove any redundant or malformed content. "
            f"If no gaps are found in a category, return an empty list for that category. "
            f"Output Format: Return a JSON object with the structure:\n"
            f"{{\"gaps\": {{\"methodological_gaps\": [{{\"description\": \"\", \"suggestion\": \"\"}}], "
            f"\"unaddressed_questions\": [{{\"description\": \"\", \"suggestion\": \"\"}}], "
            f"\"source_gaps\": [{{\"description\": \"\", \"suggestion\": \"\"}}]}}}}"
        ),
        expected_output="A clean JSON object with structured gap analysis.",
        agent=gap_formatter,
        output_file=os.path.join(OUTPUT_DIR, f"gaps_{document_id}.json")
    )

    # Create and run the crew
    crew = Crew(
        agents=[gap_analyzer, gap_formatter],
        tasks=[analyze_gaps_task, format_gaps_task],
        verbose=True
    )

    try:
        crew.kickoff()
        output_file = os.path.join(OUTPUT_DIR, f"gaps_{document_id}.json")
        json_error = clean_json_file(output_file)
        if json_error:
            logger.error(f"Failed to clean JSON file: {json_error}")
            return {"error": f"Failed to clean JSON file: {json_error}"}
        
        with open(output_file, 'r', encoding='utf-8') as f:
            result = json.load(f)
        
        # Ensure all expected keys exist, even if empty
        if not result.get("gaps"):
            result["gaps"] = {
                "methodological_gaps": [],
                "unaddressed_questions": [],
                "source_gaps": []
            }
        else:
            result["gaps"] = {
                "methodological_gaps": result["gaps"].get("methodological_gaps", []),
                "unaddressed_questions": result["gaps"].get("unaddressed_questions", []),
                "source_gaps": result["gaps"].get("source_gaps", [])
            }
        
        logger.info(f"Generated gap analysis for document_id {document_id}: {json.dumps(result, indent=2)}")
        return result
    except Exception as e:
        logger.error(f"Error running gap finder crew for document_id {document_id}: {e}")
        return {"error": str(e)}

@app.route('/gap_finder', methods=['GET', 'POST'])
def gap_finder():
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
                filename = secure_filename(file.filename)
                file_path = os.path.join(UPLOAD_DIR, filename)
                file.save(file_path)
                logger.info(f"File saved to {file_path}")
                
                cleanup_old_files()
                text = extract_text_from_pdf(file_path)
                if not text:
                    error = "Failed to extract text from PDF."
                    logger.error(f"Failed to extract text from PDF: {file_path}")
                else:
                    # Run gap finder crew
                    gaps = run_gap_finder_crew(text, document_id)
                    if "error" in gaps:
                        error = f"Error processing document: {gaps['error']}"
                        logger.error(f"Processing error for document_id {document_id}: {error}")
                    else:
                        result = {
                            "gaps": gaps["gaps"],
                            "document_id": document_id
                        }
                        # Save result with document_id
                        output_file = os.path.join(OUTPUT_DIR, f"gaps_{document_id}.json")
                        try:
                            with open(output_file, 'w', encoding='utf-8') as f:
                                json.dump(result, f, indent=4, ensure_ascii=False)
                            logger.info(f"Saved gap analysis to {output_file}")
                        except Exception as e:
                            error = f"Failed to save gap analysis file: {str(e)}"
                            logger.error(f"Error saving gap analysis to {output_file}: {e}")
    
    logger.info(f"Rendering gap_finder.html with result: {result}, error: {error}")
    return render_template('gap_finder.html', result=result or {}, error=error, generation_date=datetime.now())

@app.route('/download/gaps/<document_id>.json')
def download_gaps(document_id):
    try:
        file_path = os.path.join(OUTPUT_DIR, f"gaps_{document_id}.json")
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

if __name__ == "__main__":
    app.run(debug=True, port=5001)  # Use different port to avoid conflict with app.py