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
import uuid

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

load_dotenv()

gemini_key = os.getenv('ARYAN_GEMINI_KEY') 
if not gemini_key:
    raise ValueError("Missing Gemini API key in environment variables")

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

# Single Main Gap Analyzer Agent
gap_analyzer_agent = Agent(
    role='Research Gap Analyzer',
    goal='Analyze uploaded literature review text to identify key research gaps, under-explored areas, and opportunities for future work. Output directly as structured JSON.',
    backstory='You are an expert academic analyst specializing in literature reviews. You identify gaps by examining themes, methodologies, findings, and limitations in the provided text.',
    verbose=True,
    llm=llm_gemini,
    tools=[]
)

def run_gap_finder_crew(text: str, document_id: str) -> Dict:
    """Run the Gap Finder crew with a single main agent to analyze and format gaps."""
    cleanup_old_files()

    # Single task for the main agent to handle analysis and formatting
    analyze_gaps_task = Task(
        description=(
            f"Analyze the following literature review text to identify 5-8 major research gaps. "
            f"Focus on under-explored areas, methodological limitations, unanswered questions, and future directions. "
            f"Text: {text[:4000]}...\n\n"  # Truncate for prompt length; full text available in context
            f"Output ONLY a valid JSON object with a 'gaps' key containing an array of objects. "
            f"Each gap object must have: 'title' (concise 1-sentence summary of the gap) and 'description' (detailed explanation with evidence from text, 2-4 sentences). "
            f"Example: {{\"gaps\": [{{\"title\": \"Gap in X\", \"description\": \"Detailed explanation...\"}}]}}"
        ),
        expected_output="JSON object with 'gaps' array of objects containing 'title' and 'description' keys.",
        agent=gap_analyzer_agent,
        output_file=os.path.join(OUTPUT_DIR, f'gaps_{document_id}.json')
    )

    # Run crew with single agent and task
    try:
        crew = Crew(
            agents=[gap_analyzer_agent],
            tasks=[analyze_gaps_task],
            verbose=True
        )
        crew.kickoff()

        # Clean and validate output
        output_file = os.path.join(OUTPUT_DIR, f'gaps_{document_id}.json')
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
                # Generate unique document ID
                document_id = str(uuid.uuid4())
                filename = secure_filename(file.filename)
                file_path = os.path.join(UPLOAD_DIR, f"{document_id}_{filename}")
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
    app.run(debug=True, port=5000)  # Use different port to avoid conflict with app.py