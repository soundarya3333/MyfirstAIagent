from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel, load_tool, tool
import datetime
import requests
import pytz
import yaml
from tools.final_answer import FinalAnswerTool
from Gradio_UI import GradioUI

# -------------------------------
# Custom tools
# -------------------------------

@tool
def my_custom_tool(arg1: str, arg2: int) -> str:
    """
    A tool that currently does nothing but demonstrates a valid docstring.
    Args:
        arg1 (str): The first argument description.
        arg2 (int): The second argument description.
    Returns:
        str: A string indicating this tool is a placeholder.
    """
    return "What magic will you build?"


@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """
    A tool that fetches the current local time in a specified timezone.
    Args:
        timezone (str): A string representing a valid timezone (e.g., 'America/New_York').
    Returns:
        str: A message with the current local time in the specified timezone, or an error message.
    """
    try:
        tz = pytz.timezone(timezone)
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"


# -------------------------------
# Agent setup
# -------------------------------

final_answer = FinalAnswerTool()

# Model config
model = HfApiModel(
    max_tokens=2096,
    temperature=0.5,
    model_id='Qwen/Qwen2.5-Coder-32B-Instruct',  # fallback: possible overload
    custom_role_conversions=None,
)

# Load tools from Hub
image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

# Load system prompts
with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)

# Create the agent
agent = CodeAgent(
    model=model,
    tools=[final_answer, my_custom_tool, get_current_time_in_timezone, image_generation_tool], 
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name="smol_agent",  # valid Python identifier
    description="My first SmolAgent using multiple tools",
    prompt_templates=prompt_templates
)

# Launch the Gradio UI
GradioUI(agent).launch()
