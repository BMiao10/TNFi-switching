import regex as re
import os
import numpy as np
import pandas as pd
from bertopic import BERTopic

from transformers import AutoTokenizer, AutoModel
import torch
import json

from nltk.corpus import stopwords
from pymed.external import bertopic
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download('wordnet')

def hft_embed(model, tokenizer, docs,  padding="max_length", max_length=510, 
               embed_type="CLS", decoder=False,**kwargs):
    """
    CLS embedding values for huggingface transformers
    TODO: make a CPU only version
    
    Params:
        model (AutoModel): Huggingface tokenizer
        tokenizer (AutoTokenizer): Huggingface tokenizer
        docs (list<str>): text to embed
        padding (str): type of padding ("max_length", None)
        max_length (int): maximum token length of model
        embed_type (str): either "CLS" or "pool" for mean pooling, default "CLS"
        decoder (bool): whether the model has a decoder value
        **kwargs for tokenizer
        
    Returns:
        CLS embedding value
    """
    # add padding token if none 
    if padding is not None:
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # tokenize
    input_ids = tokenizer(docs, return_attention_mask=True, padding=padding, 
                          add_special_tokens=True, truncation=True,max_length=max_length, **kwargs)
    tokens = tokenizer.batch_encode_plus(docs, return_attention_mask=True, add_special_tokens=True, max_length=max_length, padding=padding, return_tensors="pt")
    
    # forward pass through model 
    with torch.no_grad():
        
        if decoder:
            outputs = model.encoder(**tokens, output_hidden_states=True)
            #outputs = model.encoder(input_ids=input_ids, output_hidden_states=True)
        else:
            #outputs = model(input_ids, output_hidden_states=True)
            outputs = model(**tokens, output_hidden_states=True)
            
    # get CLS or mean pool embedding
    # mean pool works well for t5 models: https://arxiv.org/abs/2108.08877
    last_layer = outputs.last_hidden_state
    
    if embed_type == "pool":
        mean_pooled = torch.mean(last_layer, dim=1)
        return mean_pooled
    else:
        cls_embeddings = last_layer[:, 0, :]
        return cls_embeddings
    
def extractBERTopics(docs, _nlp=None, seed=None, nr_topics='auto',  labels=None, embeddings=None, 
                     umap_model=None, n_gram_range=(1,1), min_topic_size=10, **kwargs):
    
    """
    Get topics using BERTopic run on spacy embeddings 
    https://arxiv.org/abs/2203.05794
    
    Params: 
        bert_df (list<str>): clinical trials data with inclusion and exclusion criteria split
        _nlp (None, spacy model): spacy model for embedding, ignored if custom embeddings are passed 
        stopwords (list): list of stopwords to remove
        criteria_col (str): criteria column in criteria_df
        cache_embeddings (str, pathlike): folder to save embeddings to for easier loading, should be in form "path/folder/"
        class_col (str): column containing groups to plot top topics for
        embeddings (np.ndarray): custom embeddings, supersedes nlp model
        nr_topics (str, int): number of topics for BERTopic to select, use 'auto' for DBSCAN auto selection
        
    """
    # set seed for reproducibility
    if seed is not None: umap_model = umap.UMAP(random_state=seed,  **kwargs)
        
    # Train our topic model using our pre-trained sentence-transformers embeddings
    model = BERTopic(nr_topics=nr_topics, umap_model=umap_model,calculate_probabilities=True).fit(docs, embeddings)
    topics, prob = model.transform(docs, embeddings)

    return model, topics, prob
    
def run_bertopic(docs,
        med_class,
        n_neighbors,
        n_components,
        min_topic_size=10,
        seed=1,
        nr_topics="auto",
                **kwargs):
    """
    docs: 
    """
    # All stopwords with "patient"
    all_stopwords = stopwords.words('english')
    all_stopwords.extend(["patient", "due", "like"])

    # Collect reasons and do light processing
    docs = [re.sub('[^a-zA-Z0-9 \n]', ' ', str(s).lower()) for s in docs]
    docs = ["none" if "reason" in s 
            else "none" if s=="nan"
            else s for s in docs]
    
    # deal with any unformatted strings
    '''
    if any(["{" in s for s in docs]):
        docs = [json.loads(_manual_json_formatting(s)) if type(s)==str else {"Type":None, "Description":None} for s in docs]
        gpt_by_encounter["gpt4_reason_cluster"] = list(pd.DataFrame.from_records(docs)["Type"])
        gpt_by_encounter["gpt4_reason"] = list(pd.DataFrame.from_records(docs)["Description"])
        docs = list(gpt_by_encounter["gpt4_reason"])
        print(gpt_by_encounter["gpt4_reason_cluster"].str.lower().value_counts())
    '''
    
    # Remove stop words
    docs = [s.split(" ") for s in docs]
    docs = [[s for s in split if s not in all_stopwords] for split in docs]
    docs = [" ".join(s) for s in docs]
    
    # Lemmatize
    wnl = WordNetLemmatizer()
    docs = [wnl.lemmatize(i) for i in docs]
    
    # Save embeddings
    #bert_tokenizer, ucsf_bert = llm.load_ucsfbert()
    ucsf_path="/wynton/group/ichs/shared_models/ucsf-bert"
    bert_tokenizer = AutoTokenizer.from_pretrained(ucsf_path, local_files_only=True)
    ucsf_bert = AutoModel.from_pretrained(ucsf_path, local_files_only=True)
    
    if not os.path.isfile(f'./data/{med_class}/gpt4/{med_class}_stop_reason_UCSFBERT_embeddings.npy'):
        print("here")
        embeddings = hft_embed(ucsf_bert, 
                               docs=docs, 
                               tokenizer=bert_tokenizer)
        np.save(f'./data/{med_class}/gpt4/{med_class}_stop_reason_UCSFBERT_embeddings.npy', embeddings)

    # =============================================================================
    # RUN BERTOPIC
    # =============================================================================

    ### BERTopic using all patient data
    # load embeddings and run BERTopic 
    embeddings = np.load(f'./data/{med_class}/gpt4/{med_class}_stop_reason_UCSFBERT_embeddings.npy')
    model, topics, all_probs = extractBERTopics(docs, 
                                                       _nlp=None, 
                                                       seed=seed, 
                                                       nr_topics=nr_topics, 
                                                       labels=None,
                                                       embeddings=embeddings, 
                                                       umap_model=None,
                                                       n_gram_range=(1,1), 
                                                       min_topic_size=min_topic_size,
                                                      **{"n_neighbors":n_neighbors,
                                                         "n_components":n_components,
                                                         "min_dist":0.0,
                                                         "metric":"euclidean"})

    print(model.topic_labels_)

    reasons_df = pd.DataFrame.from_dict({"docs":docs, "topics":topics})
    print(reasons_df["topics"].value_counts())
    
    # BERTopic
    fig = model.visualize_barchart(top_n_topics=reasons_df["topics"].nunique(), n_words=10, **kwargs)#.visualize_documents(docs, reduced_embeddings=embeddings, )
    fig.write_html(f"./figures/supplement/{med_class}_orig_topics.html")

    fig = model.visualize_hierarchy() #
    fig.write_html(f"./figures/supplement/{med_class}_orig_topic_hierarchy.html")
    
    return model, reasons_df
