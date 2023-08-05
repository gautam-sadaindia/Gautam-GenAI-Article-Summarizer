from abc import ABC, abstractmethod

class AbstractModel(ABC):    
    @abstractmethod
    def getDisplayName(self):
        pass

    @abstractmethod
    def _predictInternal(self, prompt):
        pass

    def predict(self, text: str, prompt):
        if prompt == "":
            prompt = "Provide a summary for the following article in four sentences:\n{0}"
        prompt = prompt.format(text)
        return self._predictInternal(prompt)