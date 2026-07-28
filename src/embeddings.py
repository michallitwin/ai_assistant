from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model(model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> HuggingFaceEmbeddings:
    #run model on cpu for local environment compatibility
    model_kwargs={'device': 'cpu'}
    #normalize vectors to unit length for faster cosine similarity calculation
    encode_kwargs={'normalize_embeddings': True}


    #initialize and return the HuggingFace embeddings wrapper
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
