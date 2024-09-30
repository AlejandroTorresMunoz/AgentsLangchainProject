from langchain_ollama import ChatOllama
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from src.tools_definitions import get_ticker_data, get_current_date
from langchain.prompts import MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
import datetime
from langchain.agents.format_scratchpad.openai_functions import format_to_openai_functions
from langchain_core.agents import AgentFinish


def load_llm_model(prompt):
    llm_model = ChatOllama(model="llama3.1:8b", temperature=0.1)
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", prompt),  # Descripción proporcionada al modelo
            MessagesPlaceholder(variable_name="chat_history"),  # Historial del chat
            ("human", "{input}")  # Entrada del usuario
        ]
    )
    return prompt_template | llm_model


def load_llm_tools():
    return [get_ticker_data, get_current_date]

def load_llm_chain():
    llm_model = load_llm_model().bind_tools(load_llm_tools())
    output_parser = StrOutputParser()
    prompt = ChatPromptTemplate([
        ("system", "You are a virtual assistant in the context of a financial news and recommendations site. You can execute some tools in case the user requests it. If the user's question pertains to a tool function, execute the tool. Otherwise, provide a textual response."),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    chain = prompt | llm_model | output_parser
    return chain

def run_agent(agent, user_input):
    intermediate_steps = []
    
    result = agent.invoke({
        "input": user_input,
        "agent_scratchpad": format_to_openai_functions(intermediate_steps)  # Formateo de los pasos intermedios
    })
    
    # Procesamiento del resultado
    if isinstance(result, AgentFinish):
        return result
    
    # Lógica para invocar herramientas según el resultado
    print("Here's the result")
    print(result)
    tool = {
        "get_ticker_data": get_ticker_data, 
        "get_current_date": get_current_date
    }[result.tool]
    
    observation = tool.run(result.tool_input)
    intermediate_steps.append((result, observation))  # Agregar el resultado a los pasos intermedios
    
    return observation  # Retornar la respuesta
