from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

load_dotenv() ## Load environment variables from .env file

# Define a Pydantic model to structure AI responses
class ResearchResponse(BaseModel): #All fields wanted as an output from my LLM Call
    topic: str # The main topic of research
    summary: str # Summary of the topic
    sources: list[str] # List of sources used in the research
    tools_used: list[str] # Tools used to gather information
    
    
# Initialize the AI model with OpenAI's GPT-4o-mini   
llm = ChatOpenAI(model="gpt-4o-mini")
parser = PydanticOutputParser(pydantic_object=ResearchResponse) # Set up an output parser to convert AI responses into structured format

# Define the prompt template for the AI assistant
prompt = ChatPromptTemplate.from_messages(
    [
        # System message: Defines the AI's role and response format
        (
            "system",
            """
            You are a research assistant. You will be given a topic and you will provide a summary of the topic, list the sources you used to gather information, and list the tools you used to gather information messages.
            Answer the user query and use necessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"), # Placeholder for chat history (if implementing conversation memory)
        ("human", "{query}"), # User query placeholder
        ("placeholder", "{agent_scratchpad}"), # Agent scratchpad for tool interactions
    ]
).partial(format_instructions=parser.get_format_instructions()) # Insert structured output instructions

# Define the available tools for the AI agent
tools = [search_tool, wiki_tool, save_tool]

# Create an AI agent capable of calling external tools for research
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) # Set up an executor to handle agent interactions
query = input("How can I help you ?") # Get user input as a research query
raw_response = agent_executor.invoke({"query": query}) # Invoke the AI agent with the query and retrieve raw output
print(raw_response) # Print raw AI response for debugging

try:
    structured_response = parser.parse(raw_response.get("output")[0]["text"]) #Parsing the output to get the structured response
    print(structured_response)
except Exception as e:
    print("Error parsing the output:", e, "Raw Response -", raw_response) # Handle parsing errors gracefully and print the raw response for debugging
