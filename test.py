from crewai_tools import ArxivPaperTool

tool = ArxivPaperTool(
    download_pdfs=True, 
    save_dir="./arxiv_pdfs",
)
print(tool.run(search_query="Deep learning has contributed a lot in medical imaging", max_results=3))