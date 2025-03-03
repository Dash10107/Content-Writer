import streamlit as st
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def run_crew(topic, llm_model="gpt-4", n_search=5, verbose=True):
    """
    Runs the research and content generation crew workflow.
    """
    # Setup LLM and search tool with provided options
    llm = LLM(model=llm_model)
    search_tool = SerperDevTool(n=n_search)

    # Agent 1: Senior Research Analyst (shortened for token efficiency)
    senior_research_analyst = Agent(
        role="Senior Research Analyst",
        goal=f"Research and analyze {topic} from reliable sources.",
        backstory="Expert in web research, fact-checking, and synthesizing insights.",
        allow_delegation=True,
        verbose=verbose,
        tools=[search_tool],
        llm=llm
    )

    # Agent 2: Content Writer (shortened for token efficiency)
    content_writer = Agent(
        role="Content Writer",
        goal=f"Convert research on {topic} into an engaging blog post.",
        backstory="Skilled writer who simplifies technical content while keeping it factual.",
        allow_delegation=True,
        verbose=verbose,
        llm=llm
    )

    # Task 1: Research Task
    research_task = Task(
        description=f"Research {topic}, covering recent trends, expert opinions, and statistics. Ensure sources are credible and well-organized.",
        expected_output="A structured research brief with key findings, verified facts, and citations.",
        agent=senior_research_analyst
    )

    # Task 2: Writing Task
    writing_task = Task(
        description=f"Write an engaging, accurate blog post on {topic} using the research brief. Maintain clarity and citations.",
        expected_output="A well-structured blog post with proper citations and markdown formatting.",
        agent=content_writer
    )

    # Assemble the crew with both agents and tasks
    crew = Crew(
        agents=[senior_research_analyst, content_writer],
        tasks=[research_task, writing_task],
        verbose=verbose
    )

    # Kickoff the workflow with the provided topic
    result = crew.kickoff(inputs={"topic": topic})
    return result

def main():
    # App Title and Description
    st.set_page_config(page_title="Generative AI Research & Content Creator", layout="wide")
    st.title("Generative AI Research & Content Creator")
    st.markdown(
        """
        This app uses a CrewAI workflow with two agents to first conduct research on a given topic and then generate an engaging blog post based on that research.
        """
    )

    # Sidebar options for additional settings
    st.sidebar.header("Configuration Options")
    llm_model = st.sidebar.selectbox("Choose LLM Model", options=["gpt-4", "gpt-3.5-turbo"], index=0)
    n_search = st.sidebar.slider("Number of Search Results", min_value=1, max_value=10, value=5)
    verbose = st.sidebar.checkbox("Verbose Logging", value=True)

    # Main input for topic
    topic = st.text_input("Enter a Topic", "Medical Industry Using Generative AI")

    # Button to start processing
    if st.button("Generate Content"):
        if not topic.strip():
            st.error("Please enter a valid topic.")
        else:
            with st.spinner("Generating research and content. This may take a moment..."):
                try:
                    result = run_crew(topic, llm_model=llm_model, n_search=n_search, verbose=verbose)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    return
            st.success("Content generation complete!")
            
            # Display the result in an expandable section
            with st.expander("View Generated Output", expanded=True):
                st.markdown(str(result))

            # Allow downloading the result as a text file (convert result to string)
            st.download_button(
                label="Download Output",
                data=str(result),
                file_name="generated_content.txt",
                mime="text/plain"
            )

if __name__ == '__main__':
    main()
