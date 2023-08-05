from models.abstract_model import AbstractModel
from langchain.llms import OpenAI

class LangChainModel(AbstractModel):
    def getDisplayName(self):
        return "OpenAI: LangChain"
    
    def _predictInternal(self, prompt):
        llm = OpenAI()
        return llm.predict(prompt)