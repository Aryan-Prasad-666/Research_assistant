from flask import Flask, render_template, request
import os
from dotenv import load_dotenv
from crewai import Agent, Crew, Task, LLM
from crewai_tools import SerperDevTool, ArxivPaperTool
from langchain_huggingface import HuggingFaceEmbeddings
import logging
from datetime import datetime
import json
import re

# Initialize Flask app
app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Validate environment variables
gemini_key = os.getenv('GEMINI_API_KEY')
serper_key = os.getenv('SERPER_API_KEY')
if not all([gemini_key, serper_key]):
    raise ValueError("Missing one or more API keys in environment variables (GEMINI_API_KEY, SERPER_API_KEY)")

# Initialize LLM (Gemini for all agents)
llm_gemini = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=gemini_key,
    temperature=0.6
)

# hf_embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-mpnet-base-v2",
#     model_kwargs={"device": "cpu"},
#     encode_kwargs={"normalize_embeddings": False}
# )

# Initialize Tools
serper_tool = SerperDevTool(api_key=serper_key, n_results=5)  # Limit to top 5 results
arxiv_tool = ArxivPaperTool()

# Directory for temporary files
OUTPUT_DIR = os.path.join(os.getcwd(), 'static', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Keep only the last N files to avoid filling disk
MAX_FILES = 30
def cleanup_old_files():
    """Remove old files in OUTPUT_DIR to manage disk space."""
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
                logger.error(f"Error removing file {old}: {e}")

def clean_json(response_content: str) -> str:
    """Remove ```json and ``` markers from response content and strip whitespace."""
    try:
        if not response_content.strip():
            logger.warning("Empty JSON content received")
            return "[]"
        cleaned_content = re.sub(r'^```json\s*|\s*```$', '', response_content).strip()
        json.loads(cleaned_content)  # Validate JSON
        return cleaned_content
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON after cleaning: {str(e)}")
        return "[]"
    except Exception as e:
        logger.error(f"Error cleaning JSON: {str(e)}")
        return "[]"

# Define Agents
youtube_agent = Agent(
    role='YouTube Resource Fetcher',
    goal='Fetch relevant YouTube videos, tutorials, lectures, and explainers based on the query using web search.',
    backstory='You are an expert in sourcing educational video content from YouTube using web search tools for research and learning purposes.',
    verbose=True,
    llm=llm_gemini,
    tools=[serper_tool]
)

serper_agent = Agent(
    role='Web Resource Fetcher',
    goal='Search the web for blogs, online courses (MOOCs), tutorials, documentation, and Q&A forums using Google search, returning the top 5 most relevant results.',
    backstory='You are a web researcher skilled in aggregating diverse online resources from trusted sources.',
    verbose=True,
    llm=llm_gemini,
    tools=[serper_tool]
)

arxiv_agent = Agent(
    role='arXiv Resource Fetcher',
    goal='Pull recent or highly cited research papers, abstracts, categories, and download links from arXiv.',
    backstory='You are an academic paper scout focused on scientific and technical publications.',
    verbose=True,
    llm=llm_gemini,
    tools=[arxiv_tool]
)

aggregator_agent = Agent(
    role='Resource Aggregator',
    goal='Merge and organize results from YouTube, Serper, and arXiv tasks into categorized sections, formatted as HTML lists for frontend display.',
    backstory='You are a knowledge integrator that combines multi-source data into a cohesive, display-ready structure.',
    verbose=True,
    llm=llm_gemini,
    tools=[]
)

def run_knowledge_crew(query):
    """Run the Knowledge Hub crew to fetch and aggregate resources for the given query."""
    cleanup_old_files()

    # Define Tasks with structured output expectations
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
        callback=lambda x: logger.info(f"ArXiv task output: {x}")
    )

    # Clean JSON outputs before aggregation
    def clean_task_output(file_path):
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    raw_content = f.read()
                    logger.info(f"Raw content of {file_path}: {raw_content[:200]}...")
                    cleaned_content = clean_json(raw_content)
                    json.loads(cleaned_content)  # Validate JSON
                    with open(file_path, 'w') as f:
                        f.write(cleaned_content)
                    logger.info(f"Cleaned JSON for {file_path}: {cleaned_content[:200]}...")
            else:
                logger.warning(f"File not found: {file_path}")
                with open(file_path, 'w') as f:
                    f.write('[]')  # Write empty array as fallback
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in {file_path}: {str(e)}")
            with open(file_path, 'w') as f:
                f.write('[]')  # Write empty array as fallback
        except Exception as e:
            logger.error(f"Error cleaning {file_path}: {str(e)}")
            with open(file_path, 'w') as f:
                f.write('[]')  # Write empty array as fallback

    # Run tasks and clean outputs
    crew = Crew(
        agents=[youtube_agent, serper_agent, arxiv_agent],
        tasks=[youtube_task, serper_task, arxiv_task],
        verbose=True
    )
    crew.kickoff()

    # Clean YouTube, Serper, and arXiv results before aggregation
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

    # Run aggregation
    try:
        crew = Crew(
            agents=[aggregator_agent],
            tasks=[aggregate_task],
            verbose=True
        )
        crew.kickoff()

        # Parse and clean aggregated results
        aggregated_file = os.path.join(OUTPUT_DIR, 'aggregated_results.json')
        if os.path.exists(aggregated_file):
            with open(aggregated_file, 'r') as f:
                raw_content = f.read()
                logger.info(f"Raw content of aggregated_results.json: {raw_content[:200]}...")
                cleaned_content = clean_json(raw_content)
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

if __name__ == "__main__":
    app.run(debug=True, port=5000)