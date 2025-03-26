import asyncio
from llama_index.core.agent.workflow import FunctionAgent,  AgentWorkflow
from llama_index.core.workflow import Context
from llama_index.llms.ollama import Ollama
# Para cargar variables de entorno
from dotenv import load_dotenv
import os

load_dotenv()  # busca y carga el archivo .env
endpoint2 = os.environ["URL_EXT_01"]


# Define a simple calculator tool
def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b


# Cambio el agente FuncitionAgent por un AgentWorkflow (orquestador)
# Que permite el context
workflow = AgentWorkflow.from_tools_or_functions(
    [multiply],
    llm=Ollama(model="llama3.2:latest", base_url=endpoint2, request_timeout=360.0),
    system_prompt="You are a helpful assistant that can multiply two numbers.",
)

# create context
ctx = Context(workflow)


async def main():
    # Run the agent
    response = await workflow.run("Hi, my name is Ana. What is 22 * 3?", ctx=ctx)
    print(str(response))
    response = await workflow.run("Do you remember my name?", ctx=ctx)
    print(str(response))


# Run the agent
if __name__ == "__main__":
    asyncio.run(main())