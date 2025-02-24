import warnings
warnings.filterwarnings('ignore')

# Import necessary libraries
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
import os
from utils import get_groc_api_key
import gradio as gr

# Set environment variables and initialize tools
groc_api_key = get_groc_api_key()
os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'

# tools
serper_tool = SerperDevTool()
scraper_tool = ScrapeWebsiteTool()

# Define Agents
planner = Agent(
    role="Content planner",
    goal="Plan engaging and factually accurate content on {topic}",
    backstory="You're working on planning a blog article "
              "about the topic: {topic}. "
              "You collect information that helps the "
              "audience learn something "
              "and make informed decisions. "
              "Your work is the basis for "
              "the Content Writer to write an article on this topic.",
    tools=[serper_tool, scraper_tool],
    allow_delegation=False,
    verbose=True
)

writer = Agent(
    role="Content Writer",
    goal="Write insightful and factually accurate "
         "opinion piece about the topic: {topic}",
    backstory="You're working on a writing "
              "a new opinion piece about the topic: {topic}. "
              "You base your writing on the work of "
              "the Content Planner, who provides an outline "
              "and relevant context about the topic. "
              "You follow the main objectives and "
              "direction of the outline, "
              "as provide by the Content Planner. "
              "You also provide objective and impartial insights "
              "and back them up with information "
              "provide by the Content Planner. "
              "You acknowledge in your opinion piece "
              "when your statements are opinions "
              "as opposed to objective statements.",
    tools=[serper_tool, scraper_tool],
    allow_delegation=False,
    verbose=True
)

editor = Agent(
    role="Editor",
    goal="Edit a given blog post to align with "
         "the writing style of the organization.",
    backstory="You are an editor who receives a blog post "
              "from the Content Writer. "
              "Your goal is to review the blog post "
              "to ensure that it follows journalistic best practices, "
              "provides balanced viewpoints "
              "when providing opinions or assertions, "
              "and also avoids major controversial topics "
              "or opinions when possible.",
    tools=[serper_tool, scraper_tool],
    allow_delegation=False,
    verbose=True
)

# Define Tasks
plan = Task(
    description=(
        "1. Prioritize the latest trends, key players, "
        "and noteworthy news on {topic}.\n"
        "2. Identify the target audience, considering "
        "their interests and pain points.\n"
        "3. Develop a detailed content outline including "
        "an introduction, key points, and a call to action.\n"
        "4. Include SEO keywords and relevant data or sources."
    ),
    expected_output="A comprehensive content plan document "
                    "with an outline, audience analysis, "
                    "SEO keywords, and resources.",
    agent=planner,
)

write = Task(
    description=(
        "1. Use the content plan to craft a compelling "
        "blog post on {topic}.\n"
        "2. Incorporate SEO keywords naturally.\n"
        "3. Sections/Subtitles are properly named "
        "in an engaging manner.\n"
        "4. Ensure the post is structured with an "
        "engaging introduction, insightful body, "
        "and a summarizing conclusion.\n"
        "5. Proofread for grammatical errors and "
        "alignment with the brand's voice.\n"
    ),
    expected_output="A well-written blog post "
                    "in markdown format, ready for publication, "
                    "each section should have 2 or 3 paragraphs.",
    agent=writer,
)

edit = Task(
    description=("Proofread the given blog post for "
                 "grammatical errors and "
                 "alignment with the brand's voice."),
    expected_output="A well-written blog post in markdown format, "
                    "ready for publication, "
                    "each section should have 2 or 3 paragraphs.",
    agent=editor
)

# Create the Crew
crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan, write, edit],
    verbose=True
)

# Kickoff the process
def generate_blog(topic):
    result = crew.kickoff(inputs={"topic": topic})
    return result.raw

# Interface

# ... (keep all the previous code unchanged)

# Interface

def user_interface():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(
        """
        # 🚀 AI-Powered Blog Post Generator
        
        Welcome to the future of content creation! This tool uses advanced AI to generate 
        high-quality blog posts on any topic you choose. Simply enter your desired topic, 
        and watch as our AI team of planners, writers, and editors craft a compelling article for you.
        
        ## How it works:
        1. **Plan**: Our AI planner outlines the key points and structure.
        2. **Write**: The AI writer creates engaging content based on the plan.
        3. **Edit**: Our AI editor polishes the article for grammar and style.
        
        Give it a try and revolutionize your content creation process!
        """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                topic_input = gr.Textbox(
                    lines=3, 
                    label="Enter Your Blog Topic",
                    placeholder="e.g., The Future of Artificial Intelligence in Healthcare"
                )
                generate_button = gr.Button("Generate Blog Post", variant="primary")
            
            with gr.Column(scale=1):
                gr.Markdown(
                """
                ### Tips for best results:
                - Be specific with your topic
                - Include key aspects you want covered
                - Consider your target audience
                """
                )
        
        with gr.Row():
            output = gr.TextArea(label="Generated Blog Post", lines=30, max_lines=50)
        
        generate_button.click(
            generate_blog, 
            inputs=topic_input, 
            outputs=output,
            show_progress="full"
        )
        
        gr.Markdown(
        """
        ### About this tool
        This blog post generator uses CrewAI and GPT-3.5-turbo to create content. 
        While the output is high-quality, always review and fact-check the generated content before publishing.
        """
        )
    
    return demo

# Launch the interface
if __name__ == "__main__":
    demo = user_interface()
    demo.launch()