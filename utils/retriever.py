import faiss
import numpy as np

class VectorStore:
    def __init__(self, embeddings, texts):
        self.texts = texts
        dimension = len(embeddings[0])
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype("float32"))

    def search(self, query_embedding, k=3):
        distances, indices = self.index.search(
            np.array([query_embedding]).astype("float32"), k
        )

        return [self.texts[i] for i in indices[0]]