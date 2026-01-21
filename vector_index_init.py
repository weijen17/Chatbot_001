
import os
from dotenv import load_dotenv
import argparse
from src.config.settings import settings
from src.data import faiss_index_setup
import pandas as pd
from sentence_transformers import SentenceTransformer

EMBED_MODEL = settings.EMBED_MODEL
INPUT_NAME = settings.INPUT_NAME

if __name__ == "__main__":
    embedder = SentenceTransformer(EMBED_MODEL)
    df1=pd.read_excel(INPUT_NAME)
    df1=df1.reset_index(drop=True).reset_index()
    data = df1[['index','_id','author','user_followers_count','engagement','title_body']].values.tolist()
    faiss_index_setup(embedder, data)

