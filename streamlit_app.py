import streamlit as st
import time
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv

load_dotenv()

def run_crew_with_logging(topic, llm_model="gpt-4", n_search=5, verbose=True, log_callback=None):
    """
    Runs the Crew workflow and sends log messages via the log_callback.
    If log_callback is provided, it should be a function that accepts a string.
    """
    # Setup LLM and search tool with provided options
    llm = LLM(model=llm_model)
    search_tool = SerperDevTool(n=n_search)

    # Log the start of each major step
    if log_callback: log_callback("Initializing Senior Research Analyst...")
    senior_research_analyst = Agent(
        role="Senior Research Analyst",
        goal=f"Research and analyze {topic} from reliable sources.",
        backstory="Expert in web research, fact-checking, and synthesizing insights.",
        allow_delegation=True,
        verbose=verbose,
        tools=[search_tool],
        llm=llm
    )

    if log_callback: log_callback("Initializing Content Writer...")
    content_writer = Agent(
        role="Content Writer",
        goal=f"Convert research on {topic} into an engaging blog post.",
        backstory="Skilled writer who simplifies technical content while keeping it factual.",
        allow_delegation=True,
        verbose=verbose,
        llm=llm
    )

    if log_callback: log_callback("Creating research task...")
    research_task = Task(
        description=f"Research {topic}, covering recent trends, expert opinions, and statistics. Ensure sources are credible and well-organized.",
        expected_output="A structured research brief with key findings, verified facts, and citations.",
        agent=senior_research_analyst
    )

    if log_callback: log_callback("Creating writing task...")
    writing_task = Task(
        description=f"Write an engaging, accurate blog post on {topic} using the research brief. Maintain clarity and citations.",
        expected_output="A well-structured blog post with proper citations and markdown formatting.",
        agent=content_writer
    )

    if log_callback: log_callback("Assembling the crew...")
    crew = Crew(
        agents=[senior_research_analyst, content_writer],
        tasks=[research_task, writing_task],
        verbose=verbose
    )

    if log_callback: log_callback("Kicking off the crew workflow...")
    result = crew.kickoff(inputs={"topic": topic})
    if log_callback: log_callback("Workflow complete.")
    return result

def main():
    st.set_page_config(page_title="Generative AI Research & Content Creator", layout="wide")
    st.title("Generative AI Research & Content Creator")
    st.markdown(
        """
        This app uses a CrewAI workflow to first conduct research on a given topic and then generate an engaging blog post.
        """
    )

    # Sidebar configuration
    st.sidebar.header("Configuration Options")
    llm_model = st.sidebar.selectbox("Choose LLM Model", options=["gpt-4", "gpt-3.5-turbo"], index=0)
    n_search = st.sidebar.slider("Number of Search Results", min_value=1, max_value=10, value=5)
    verbose = st.sidebar.checkbox("Verbose Logging", value=True)

    # Main input
    topic = st.text_input("Enter a Topic", "Medical Industry Using Generative AI")

    # Placeholder for logs
    log_placeholder = st.empty()
    logs = []

    def log_callback(message):
        logs.append(message)
        log_placeholder.text("\n".join(logs))

    # Button to start processing
    if st.button("Generate Content"):
        if not topic.strip():
            st.error("Please enter a valid topic.")
        else:
            logs.clear()  # Clear previous logs
            log_placeholder.text("Starting process...")
            with st.spinner("Generating research and content. This may take a moment..."):
                try:
                    result = run_crew_with_logging(
                        topic,
                        llm_model=llm_model,
                        n_search=n_search,
                        verbose=verbose,
                        log_callback=log_callback
                    )
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    return
            st.success("Content generation complete!")
            
            # Display the generated content
            with st.expander("View Generated Output", expanded=True):
                st.markdown(str(result))
            # Download button for the generated content
            st.download_button(
                label="Download Output",
                data=str(result),
                file_name="generated_content.txt",
                mime="text/plain"
            )

if __name__ == '__main__':
    main()
    