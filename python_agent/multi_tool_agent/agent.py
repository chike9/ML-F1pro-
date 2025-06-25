from google.adk.agents import Agent
from dotenv import load_dotenv
load_dotenv()
def explain_code(code: str) -> dict:
    """Explains what a Python code snippet does in simple terms"""
    return {
        "status": "success",
        "explanation": f"This tool would explain the code: {code}"
    }

def fix_code(code: str) -> dict:
    """Finds and fixes errors in Python code"""
    return {
        "status": "success",
        "fixed_code": f"# Fixed version of:\n{code}",
        "changes": "Fixed syntax errors"
    }

root_agent = Agent(
    name="simple_python_coder",
    model="gemini-2.0-flash",
    description="Helps with Python coding tasks",
    instruction=(
        "You are a Python coding assistant. "
        "Use your tools to explain or fix code snippets."
    ),
    tools=[explain_code, fix_code]
)