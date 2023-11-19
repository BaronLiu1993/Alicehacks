
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import os
import pickle
import csv
import time
import faiss
import glob
from pprint import pprint
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
from torch import nn
from PIL import Image
from whisper_mic.whisper_mic import WhisperMic
import gc
import streamlit as st
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events
from streamlit_feedback import streamlit_feedback
from langsmith import Client

image = Image.open('Tree.png')

st.title('SurgerySim Semantic Search Tool ❤️')
embedder = SentenceTransformer('msmarco-distilbert-base-dot-prod-v3')

for ix,csv in enumerate(glob.glob('clean_comm_use.csv')):
    if ix==0:
        data=pd.read_csv(csv)
    else:
        temp=pd.read_csv(csv)
        data=pd.concat([data,temp],axis=0).reset_index(drop=True)

data=data.dropna(subset=['title','abstract']).reset_index(drop=True)
data['abstract']=data['abstract'].apply(lambda x: x.replace('\n',' ')[9:].strip())
data['text']=data['text'].apply(lambda x: x.replace('\n',' ')[12:].strip())

class faiss_search:
    
    def __init__(self,max_corpus_size = 100000,embedding_size= 768,top_k_hits= 5):
        
        self.embedding_cache_path = 'abstract-embeddings-{}-size-{}.pkl'.format('msmarco-distilbert-base-dot-prod-v3', max_corpus_size)
        n_clusters = round(np.sqrt(data.shape[0])*4)
        quantizer = faiss.IndexFlatIP(embedding_size)
        self.index = faiss.IndexIVFFlat(quantizer, embedding_size, n_clusters, faiss.METRIC_INNER_PRODUCT)
        self.index.nprobe = 3
        
    
    def embed_corpus(self,data,embedder):
        if not os.path.exists(self.embedding_cache_path):
            corpus_sentences = data['abstract'].values.tolist()
            print("Encode the corpus. This might take a while")
            corpus_embeddings = embedder.encode(corpus_sentences, show_progress_bar=True, convert_to_numpy=True)

            print("Store file on disc")
            with open(self.embedding_cache_path, "wb") as fOut:
                pickle.dump({'sentences': corpus_sentences, 'embeddings': corpus_embeddings}, fOut)
        else:
            print("Load pre-computed embeddings from disc")
            with open(self.embedding_cache_path, "rb") as fIn:
                cache_data = pickle.load(fIn)
                corpus_sentences = cache_data['sentences']
                corpus_embeddings = cache_data['embeddings']
        return corpus_sentences,corpus_embeddings
    def index_data(self,corpus_sentences,corpus_embeddings):
        print("Start creating FAISS index")
        corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1)[:, None]
        self.index.train(corpus_embeddings)
        self.index.add(corpus_embeddings)

        print("Corpus loaded with {} sentences / embeddings".format(len(corpus_sentences)))

faiss_obj=faiss_search()
corpus_sentences,corpus_embeddings=faiss_obj.embed_corpus(data,embedder)
faiss_obj.index_data(corpus_sentences,corpus_embeddings)


search_term = st.text_input(
    label=":blue[Search Your Query]",
    placeholder = "Search with...",
)

search_button = st.button(label = "Run", type = 'primary')

if search_term:
     if search_button:
        top_k_hits=5
        query='tell me about death rates of covid case'

        start_time = time.time()
        title_embedding = embedder.encode(query)

        title_embedding = title_embedding / np.linalg.norm(title_embedding)
        title_embedding = np.expand_dims(title_embedding, axis=0)

        distances, corpus_ids = faiss_obj.index.search(title_embedding, top_k_hits)

        hits = [{'corpus_id': id, 'score': score} for id, score in zip(corpus_ids[0], distances[0])]
        hits = sorted(hits, key=lambda x: x['score'], reverse=True)
        end_time = time.time()

        st.write("Input title:", query)
        st.write("\n")
        st.write("Results (after {:.3f} seconds):".format(end_time-start_time))
        st.write("\n")
        for hit in hits[0:top_k_hits]:
            st.write("\t{:.3f}\t{}".format(hit['score'], corpus_sentences[hit['corpus_id']]))


        correct_hits = util.semantic_search(title_embedding, corpus_embeddings, top_k=top_k_hits)[0]
        correct_hits_ids = set([hit['corpus_id'] for hit in correct_hits])

        ann_corpus_ids = set([hit['corpus_id'] for hit in hits])
        if len(ann_corpus_ids) != len(correct_hits_ids):
            st.write("Approximate Nearest Neighbor returned a different number of results than expected")

        recall = len(ann_corpus_ids.intersection(correct_hits_ids)) / len(correct_hits_ids)
        st.write("\nApproximate Nearest Neighbor Recall@{}: {:.2f}".format(top_k_hits, recall * 100))

        if recall < 1:
            print("Missing results:")
            for hit in correct_hits[0:top_k_hits]:
                if hit['corpus_id'] not in ann_corpus_ids:
                    print("\t{:.3f}\t{}".format(hit['score'], corpus_sentences[hit['corpus_id']]))

        gc.collect()
        del faiss_obj,corpus_sentences,corpus_embeddings