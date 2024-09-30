from src.load_model import load_llm_model
from src.agents_methods import Agent

# Suponiendo que ya tienes las herramientas necesarias
prompt = '''Answer the following questions as best you can. 
    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take. If there's no action to be executed, just return a textual answer based on the Question and Thought in the Final Answer.
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    '''.strip()

llm_model = load_llm_model(prompt)
agent = Agent(system=prompt, chat_model=llm_model)

# Ejecutar la llamada al agente
result = agent("Hey, how are you?")
print("La salida generada es la siguiente : ")
print(result)
