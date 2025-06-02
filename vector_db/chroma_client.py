import chromadb

class ChromaVectorDB(object):
    def __init__(self, collection_name="contexts"):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(collection_name)

    def add_texts(self, texts, metadata=None, ids=None):
        """
        texts: List[str] of context passages
        metadata: List[dict], can store extra info per passage
        ids: List[str] of unique IDs for each passage (optional)
        """
        if ids:
            self.collection.add(documents=texts, metadatas=metadata, ids=ids)
        else:
            self.collection.add(documents=texts, metadatas=metadata)

    def query(self, query_texts, embedding_fn, top_k=3):
        """
        query_text: str
        embedding_fn: function that takes [query_text] and returns embeddings
        top_k: how many results to return
        Returns a dict with keys "ids" and "documents", each a list-of-lists
        """
        query_embedding = embedding_fn([query_texts])
        results = self.collection.query(
            query_embeddings = [query_embedding[0]],
            n_results = top_k
        )

        return results