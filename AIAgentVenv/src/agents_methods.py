from typing import Any
from langchain_core.messages import HumanMessage, AIMessage

class Agent:
    def __init__(self, system: str, chat_model: Any):
        self.system = system
        self.chat_model = chat_model
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": self.system})

    def __call__(self, message: str):
        # self.messages.append({"role": "user", "content": message})
        # print("Messages before invoke:", self.messages)  # Para depuración
        result = self.execute(message)
        return result

    def execute(self, message):
        # Invocar el modelo de chat
        content = self.chat_model.invoke({
            "input" : message,
            "chat_history" : self.messages
        })
        content = content.content
        final_answer = None
        for line in content.split('\n'):
            if line.startswith('Final Answer:'):
                final_answer = line[len('Final Answer: '):].strip()  # Extraer el texto después de "Final Answer:"
                break  # Salir del bucle una vez que se encuentre

        if final_answer is not None:
            self.messages.append(HumanMessage(content=message))
            self.messages.append(AIMessage(content=final_answer))
            return final_answer
        else:
            return "Error executing LLM."
        
        