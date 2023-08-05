from models.abstract_model import *
from vertexai.language_models import TextGenerationModel

class TextBisonModel(AbstractModel):
    def getDisplayName(self):
        return "VertexAI: TextBison@001"

    def _predictInternal(self, prompt):
        model = TextGenerationModel.from_pretrained("text-bison@001")
        parameters = {"temperature": 0.5}
        return model.predict(prompt=prompt, **parameters).text