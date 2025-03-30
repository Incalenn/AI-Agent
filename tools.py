from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime

# ----------------------
# Custom Tool: Save to Text File
# ----------------------
def save_to_txt(data: str, filename: str= "research_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") # Get the current timestamp
    formatted_text = f"--- Research Output---\nTimestamp: {timestamp}\n\n{data}\n\n" # Format the data with a header and timestamp
    
    with open(filename, "a", encoding="utf-8") as f: # Open the file in append mode and write the formatted text
        f.write(formatted_text)
        
    return f"Data has been saved to {filename}"

save_tool = Tool(  # Define the custom save tool for AI agents
    name="save_text_to_file",
    func=save_to_txt,
    description="Save the data researched throughout the web to a text file",
)

# ----------------------
# Web Search Tool: DuckDuckGo
# ----------------------
search = DuckDuckGoSearchRun() # Initialize DuckDuckGo search tool
search_tool = Tool(
    name="search",
    func=search.run, # Function to execute web search
    description="Scrap the web for information",
)

# ----------------------
# Wikipedia Search Tool
# ----------------------
# Configure Wikipedia API wrapper to return the top result with limited content
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper) # Create a Wikipedia query tool using the API wrapper