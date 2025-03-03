from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv

load_dotenv()

topic = "Medical Industry Using Generative AI"

# Use a more efficient model
llm = LLM(model="gpt-4")  # Consider using "gpt-3.5-turbo" if needed

# Limit search results to reduce context length
search_tool = SerperDevTool(n=5)  

# Agent 1: Research Analyst (Shortened)
senior_research_analyst = Agent(
    role="Senior Research Analyst",
    goal=f"Research and analyze {topic} from reliable sources.",
    backstory="Expert in web research, fact-checking, and synthesizing insights.",
    allow_delegation=True,
    verbose=True,
    tools=[search_tool],
    llm=llm
)

# Agent 2: Content Writer (Shortened)
content_writer = Agent(
    role="Content Writer",
    goal=f"Convert research on {topic} into an engaging blog post.",
    backstory="Skilled writer who simplifies technical content while keeping it factual.",
    allow_delegation=True,
    verbose=True,
    llm=llm
)

# Task 1: Research (Shortened)
research_task = Task(
    description=f"Research {topic}, covering recent trends, expert opinions, and statistics. Ensure sources are credible and well-organized.",
    expected_output="A structured research brief with key findings, verified facts, and citations.",
    agent=senior_research_analyst
)

# Task 2: Writing (Shortened)
writing_task = Task(
    description=f"Write an engaging, accurate blog post on {topic}, using the research brief. Maintain clarity and citations.",
    expected_output="A well-structured blog post with proper citations and markdown formatting.",
    agent=content_writer
)

# Sequential Execution
crew = Crew(
    agents=[senior_research_analyst, content_writer],
    tasks=[research_task, writing_task],
    verbose=True
)

result = crew.kickoff(inputs={"topic": topic})

print(result)
