from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def get_embedding(self, sentence):
        if isinstance(sentence, str):
            sentence = [sentence]
        return self.model.encode(sentence)