from flask import Flask, render_template, request, jsonify
import json
import os
import logging
import re
from datetime import datetime
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
app = Flask(__name__)

# Retrieve API keys
serper_api_key = os.getenv('SERPER_API_KEY')
gemini_api_key = os.getenv('GEMINI_API_KEY')
if not serper_api_key:
    logger.error("SERPER_API_KEY not found in environment variables")
    raise EnvironmentError("SERPER_API_KEY is required")
if not gemini_api_key:
    logger.error("GEMINI_API_KEY not found in environment variables")
    raise EnvironmentError("GEMINI_API_KEY is required")

# Initialize LLM
llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.7,
    api_key=gemini_api_key
)

# Initialize SerperDevTool
serper_tool = SerperDevTool(
    api_key=serper_api_key,
    n_results=20  # Increased to improve metadata quality
)

# Define citation formats
CITATION_FORMATS = {
    'APA': '{author}. ({year}). {title}. {journal}, {volume}({issue}), {pages}. {doi}',
    'MLA': '{author}. "{title}." {journal}, vol. {volume}, no. {issue}, {year}, pp. {pages}. {doi}',
    'IEEE': '{author}, "{title}," {journal}, vol. {volume}, no. {issue}, pp. {pages}, {year}.',
    'Chicago': '{author}. {year}. "{title}." {journal} {volume}, no. {issue}: {pages}. {doi}.'
}

# Pydantic model for citation validation
class Citation(BaseModel):
    name: str = Field(..., min_length=1, description="Title of the source")
    citation: str = Field(..., min_length=1, description="Formatted citation string")
    author: str = Field(..., min_length=1, description="Author(s) of the source")
    title: str = Field(..., min_length=1, description="Title of the source (same as name)")
    journal: str = Field(..., min_length=1, description="Journal name")
    volume: str = Field(..., min_length=1, description="Volume number")
    issue: str = Field(..., min_length=1, description="Issue number")
    pages: str = Field(..., min_length=1, description="Page range")
    year: str = Field(..., pattern=r'^\d{4}$', description="Publication year (4 digits)")
    doi: Optional[str] = Field(None, description="DOI, if available")
    error: Optional[str] = Field(None, description="Error message for incomplete metadata")

class CitationOutput(BaseModel):
    citations: List[Citation]

# Create citation generator agent
citation_generator = Agent(
    role="Citation Generator",
    goal="Generate accurate citations for research claims, prioritizing peer-reviewed journal articles with complete metadata (author, title, journal, volume, issue, pages, year, DOI).",
    backstory="An expert in academic citation formats, skilled at finding and formatting references from web searches, focusing on high-quality peer-reviewed journal articles.",
    tools=[serper_tool],
    verbose=True,
    llm=llm
)

def clean_json_content(content: str) -> str:
    """Clean JSON content by removing markdown fences and handling malformed input."""
    content = content.strip()
    # Remove ```json, ```, or any markdown fences with case-insensitive matching
    content = re.sub(r'^```(?:json)?\s*\n', '', content, flags=re.MULTILINE | re.IGNORECASE)
    content = re.sub(r'\n```$', '', content, flags=re.MULTILINE)
    # Remove any non-JSON leading/trailing text
    try:
        # Find the first valid JSON array
        start_idx = content.find('[')
        end_idx = content.rfind(']')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            content = content[start_idx:end_idx + 1]
        content = content.strip()
        # Validate as JSON to catch early errors
        json.loads(content)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse JSON, attempting to extract valid portion: {content[:100]}...")
        # Fallback: try to extract any valid JSON-like structure
        match = re.search(r'\[\s*{.*}\s*\]', content, flags=re.DOTALL)
        if match:
            content = match.group(0)
        else:
            content = '[]'  # Return empty list if no valid JSON found
    logger.debug(f"Cleaned JSON content: {content[:100]}...")
    return content

@app.route('/citation_generator', methods=['GET', 'POST'])
def citation_generator_route():
    error = None
    result = {'citations': []}
    generation_date = None
    form_submitted = False

    if request.method == 'POST':
        try:
            # Extract form data
            claim = request.form.get('claim', '').strip()
            keywords = request.form.get('keywords', '').strip()
            citation_style = request.form.get('citation_style', 'All Styles')
            max_citations = request.form.get('max_citations', '3')
            source_type = request.form.get('source_type', 'All')
            year_start = request.form.get('year_start', '').strip()
            year_end = request.form.get('year_end', '').strip()

            # Validate required fields
            if not claim:
                error = "Claim is required."
                logger.warning("Form submitted without a claim")
                return render_template('citation_generator.html', error=error, result=result, 
                                    generation_date=generation_date, form_submitted=True, 
                                    current_year=datetime.now().year)

            # Validate max_citations
            try:
                max_citations = int(max_citations)
                if max_citations < 1 or max_citations > 10:
                    error = "Maximum citations must be between 1 and 10."
                    logger.warning(f"Invalid max_citations value: {max_citations}")
                    return render_template('citation_generator.html', error=error, result=result, 
                                        generation_date=generation_date, form_submitted=True, 
                                        current_year=datetime.now().year)
            except ValueError:
                error = "Maximum citations must be a valid number."
                logger.warning(f"Invalid max_citations value: {max_citations}")
                return render_template('citation_generator.html', error=error, result=result, 
                                    generation_date=generation_date, form_submitted=True, 
                                    current_year=datetime.now().year)

            # Validate year range
            if year_start and not year_start.isdigit():
                error = "Year Start must be a valid 4-digit year (e.g., 2010)."
                logger.warning(f"Invalid year_start: {year_start}")
                return render_template('citation_generator.html', error=error, result=result, 
                                    generation_date=generation_date, form_submitted=True, 
                                    current_year=datetime.now().year)
            if year_end and not year_end.isdigit():
                error = "Year End must be a valid 4-digit year (e.g., 2025)."
                logger.warning(f"Invalid year_end: {year_end}")
                return render_template('citation_generator.html', error=error, result=result, 
                                    generation_date=generation_date, form_submitted=True, 
                                    current_year=datetime.now().year)
            if year_start and year_end and int(year_start) > int(year_end):
                error = "Year Start must be less than or equal to Year End."
                logger.warning(f"Year range invalid: {year_start} > {year_end}")
                return render_template('citation_generator.html', error=error, result=result, 
                                    generation_date=generation_date, form_submitted=True, 
                                    current_year=datetime.now().year)

            # Construct search query
            search_query = f"{claim} peer-reviewed journal article scholarly"
            if keywords:
                search_query += f" {keywords}"
            if source_type != 'All':
                search_query += f" filetype:{source_type.lower()}"
            if year_start and year_end:
                search_query += f" after:{year_start} before:{year_end}"
            elif year_start:
                search_query += f" after:{year_start}"
            elif year_end:
                search_query += f" before:{year_end}"

            # Define output file
            output_file = 'citations.json'

            # Create task for citation generation
            citation_formats = [CITATION_FORMATS[style] for style in CITATION_FORMATS] if citation_style == 'All Styles' else [CITATION_FORMATS.get(citation_style, CITATION_FORMATS['APA'])]
            task_description = (
                f"Search for peer-reviewed journal articles to support the claim: '{claim}'. "
                f"Use the search query: '{search_query}'. "
                f"Generate exactly {max_citations} citations in the {citation_style} style(s): {', '.join(citation_formats)}. "
                f"Each citation MUST include 'name', 'citation', 'author', 'title', 'journal', 'volume', 'issue', 'pages', 'year', and 'doi' (if available). "
                f"Reject sources missing 'journal', 'volume', 'issue', or 'pages' unless no journal articles with complete metadata are found; in such cases, include an 'error' field (e.g., 'Missing journal, volume, issue, pages'). "
                f"Prioritize recent (2023–2025), high-quality journal articles from reputable sources (e.g., PubMed, IEEE, Springer) with complete metadata. "
                f"Return a JSON list of citations, each with fields: 'name' (string, source title), 'citation' (string, formatted citation), 'author' (string), 'title' (string, same as name), 'journal' (string), 'volume' (string), 'issue' (string), 'pages' (string), 'year' (string, 4 digits), 'doi' (string or null), 'error' (string or null). "
                f"Ensure the output is a valid JSON string without markdown code fences or extra text, strictly adhering to the schema: {json.dumps([{'name': 'string', 'citation': 'string', 'author': 'string', 'title': 'string', 'journal': 'string', 'volume': 'string', 'issue': 'string', 'pages': 'string', 'year': 'string', 'doi': 'string | null', 'error': 'string | null'}])}"
            )

            search_task = Task(
                description=task_description,
                expected_output=f"A JSON list of exactly {max_citations} citations with fields: name, citation, author, title, journal, volume, issue, pages, year, doi, error, prioritizing journal articles with complete metadata.",
                agent=citation_generator,
                output_file=output_file
            )

            # Initialize Crew
            crew = Crew(
                agents=[citation_generator],
                tasks=[search_task],
                verbose=True
            )

            # Execute task with retry mechanism
            @retry(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=4, max=10),
                retry=retry_if_exception_type(Exception),
                after=lambda retry_state: logger.debug(f"Retry attempt {retry_state.attempt_number} failed with {retry_state.outcome.exception()}")
            )
            def execute_crew():
                result = crew.kickoff()
                if not result:
                    raise ValueError("Crew execution returned no result")
                return result

            try:
                execute_crew()
            except Exception as e:
                logger.error(f"Crew execution failed after retries: {str(e)}")
                error = "The citation generation service is temporarily unavailable. Please try again later."
                return render_template('citation_generator.html', error=error, result=result, 
                                    generation_date=generation_date, form_submitted=True, 
                                    current_year=datetime.now().year)

            # Process the output
            try:
                if os.path.exists(output_file):
                    with open(output_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()

                    # Save raw output for debugging
                    raw_output_file = 'citations_raw.txt'
                    with open(raw_output_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    logger.debug(f"Raw output saved to {raw_output_file}: {content[:100]}...")

                    # Clean JSON content
                    clean_content = clean_json_content(content)
                    if not clean_content:
                        error = "Output file is empty after cleaning."
                        logger.error(error)
                        return render_template('citation_generator.html', error=error, result=result, 
                                            generation_date=generation_date, form_submitted=True, 
                                            current_year=datetime.now().year)

                    # Parse JSON
                    try:
                        citations_data = json.loads(clean_content)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing JSON from {output_file}: {e}")
                        error = f"Error processing citations: {e}"
                        return render_template('citation_generator.html', error=error, result=result, 
                                            generation_date=generation_date, form_submitted=True, 
                                            current_year=datetime.now().year)

                    # Validate citations data
                    if not isinstance(citations_data, list):
                        error = "Citations data is not a valid JSON list."
                        logger.error(f"Invalid citations data: {citations_data}")
                        return render_template('citation_generator.html', error=error, result=result, 
                                            generation_date=generation_date, form_submitted=True, 
                                            current_year=datetime.now().year)

                    # Validate citations with Pydantic
                    valid_citations = []
                    for citation in citations_data:
                        try:
                            validated_citation = Citation(**citation)
                            valid_citations.append({
                                'name': validated_citation.name,
                                'citation': validated_citation.citation,
                                'author': validated_citation.author,
                                'title': validated_citation.title,
                                'journal': validated_citation.journal,
                                'volume': validated_citation.volume,
                                'issue': validated_citation.issue,
                                'pages': validated_citation.pages,
                                'year': validated_citation.year,
                                'doi': validated_citation.doi,
                                'error': validated_citation.error
                            })
                        except ValidationError as e:
                            logger.warning(f"Skipping invalid citation: {citation}, error: {e}")
                            continue

                    # Filter citations with complete metadata first
                    complete_citations = [c for c in valid_citations if not c['error']]
                    incomplete_citations = [c for c in valid_citations if c['error']]
                    result['citations'] = complete_citations + incomplete_citations[:max_citations - len(complete_citations)][:max_citations]

                    # Save cleaned JSON
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(result['citations'], f, indent=2)
                    logger.debug(f"Saved cleaned JSON to {output_file}")

                    if not result['citations']:
                        error = "No valid citations found with complete metadata matching your criteria."
                        logger.warning(error)
                    elif not complete_citations:
                        error = "No citations with complete metadata found; some fields may be missing."
                        logger.warning(error)

                    # Set generation date
                    generation_date = datetime.now()
                else:
                    error = f"Output file {output_file} not found."
                    logger.error(error)
            except Exception as e:
                logger.error(f"Error processing {output_file}: {e}")
                error = f"Error processing citations: {e}"

            form_submitted = True
        except Exception as e:
            logger.error(f"Error fetching citations: {e}")
            error = "An unexpected error occurred during citation generation. Please try again later."

    return render_template('citation_generator.html', error=error, result=result, 
                         generation_date=generation_date, form_submitted=form_submitted, 
                         current_year=datetime.now().year)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)