import os
import json
import logging
from typing import Dict, List, Optional
from flask import Flask, render_template, request, jsonify, send_file
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from PyPDF2 import PdfReader
from datetime import datetime
import io
import re

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

# Initialize LLM
llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=gemini_key,
    temperature=0.7
)

# Initialize text splitter and embeddings
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Directory for vector database and output files
OUTPUT_DIR = os.path.join(os.getcwd(), 'static', 'summaries')
os.makedirs(OUTPUT_DIR, exist_ok=True)
VECTOR_DB_DIR = os.path.join(os.getcwd(), 'vector_db')
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

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
def extract_text_from_pdf(file_stream):
    """Extract text from a PDF file stream."""
    try:
        pdf_reader = PdfReader(file_stream)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
        logger.info("Successfully extracted text from PDF")
        return text.strip()
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
        # Retrieve relevant chunks
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        chunks = retriever.get_relevant_documents("summarize the key points, methodologies, and findings")
        
        # Combine chunks for summarization
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
        # Retrieve relevant chunks
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        chunks = retriever.get_relevant_documents("analyze tone, framing, and sources")
        
        # Combine chunks for bias detection
        combined_text = "\n".join([chunk.page_content for chunk in chunks])
        prompt = (
            f"Analyze the following document content for potential biases in tone, framing, and source diversity. "
            f"Return a JSON object with sections for 'tone', 'framing', and 'source_diversity'. "
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

@app.route('/', methods=['GET', 'POST'])
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
                cleanup_old_files()
                text = extract_text_from_pdf(file)
                if not text:
                    error = "Failed to extract text from PDF."
                    logger.error("Failed to extract text from uploaded PDF")
                else:
                    # Create vector store
                    vector_store = create_vector_store(text, document_id)
                    
                    # Generate summary and bias analysis
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
                        # Save result to file
                        output_file = os.path.join(OUTPUT_DIR, f"summary_{document_id}.json")
                        try:
                            with open(output_file, 'w', encoding='utf-8') as f:
                                json.dump(result, f, indent=4, ensure_ascii=False)
                            logger.info(f"Saved summary to {output_file}")
                        except Exception as e:
                            error = f"Failed to save summary file: {str(e)}"
                            logger.error(f"Error saving summary to {output_file}: {e}")
    
    return render_template('summarize.html', result=result or {}, error=error, generation_date=datetime.now())

@app.route('/download/summary/<document_id>.json')
def download_summary(document_id):
    try:
        file_path = os.path.join(OUTPUT_DIR, f"summary_{document_id}.json")
        logger.info(f"Attempting to download file: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return jsonify({"error": f"Summary file for document ID {document_id} not found"}), 404
        
        # Check file permissions
        if not os.access(file_path, os.R_OK):
            logger.error(f"No read permission for file: {file_path}")
            return jsonify({"error": f"No read permission for summary file {document_id}"}), 403
        
        # Log file size for debugging
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

if __name__ == "__main__":
    app.run(debug=True)