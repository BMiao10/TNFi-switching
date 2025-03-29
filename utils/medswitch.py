import os
import glob
import json
import time

import pandas as pd
import regex as re
import numpy as np
import tiktoken
from openai import AzureOpenAI

from sklearn.model_selection import train_test_split

### OPENAI PARAMS
keys = os.environ["OPENAI_KEYS"]
    
#openai.api_key = keys["OPENAI_API_KEY"] #os.getenv("OPENAI_API_KEY")
client = AzureOpenAI(
    api_key=keys["OPENAI_API_KEY"],  
    api_version="2023-05-15",
    azure_endpoint = keys['OPENAI_API_BASE']
    )

def get_first_or_last_values(data_df,
                              datetime_col="startdatekeyvalue", 
                              get_first=True, 
                              groupby="patientdurablekey",
                             prefix=None):
    '''
    Gets first or last value for each group when sorted by datetime_col 
    '''
    data_df = data_df.sort_values(datetime_col, ascending=get_first) # gets first occurrence if get_first is True
    data_df = data_df.groupby(groupby, sort=False).first()
    if prefix is not None:
        data_df.columns = [str(prefix)+s for s in data_df.columns]
    return data_df

def fill_prev_next_meds(ground_truth_df, med_col = "mapped_med_generic_clean"):
    """
    Helper class for medication table querying to get the previous and next medications
    """
    ## Get ground truth values for medication encounters
    ground_truth_df = ground_truth_df.sort_values(["patientdurablekey", "startdatekeyvalue"])
    ground_truth_df["next_medication"] = ground_truth_df.groupby("patientdurablekey", sort=False)[med_col].shift(periods=-1, freq=None, axis=0)
    ground_truth_df["prev_medication"] = ground_truth_df.groupby("patientdurablekey", sort=False)[med_col].shift(periods=1, freq=None, axis=0)
    ground_truth_df["curr_med_change"] = [False if p is None
                                       else False if type(p)==float
                                       else c!=p for c,p in zip(ground_truth_df[med_col], ground_truth_df["prev_medication"])]
    ground_truth_df["next_med_change"] = [False if n is None
                                       else False if type(n)==float
                                       else c!=n for c,n in zip(ground_truth_df[med_col], ground_truth_df["next_medication"])]
    
    #ground_truth_df["prev_end_date"] = ground_truth.groupby("patientdurablekey", sort=False)["enddatekeyvalue"].shift(periods=1, freq=None, axis=0)

    return ground_truth_df

def unique_trajectory(med_trajectory):
    """
    Given a list of medications, get the unique trajectory
    Eg. ["etanercept", "baricitinib", "baricitinib", "etanercept"] -> ["etanercept", "baricitinib", "etanercept"]
    """
    
    return [med_trajectory[i] for i in range(len(med_trajectory)) if (i==0) or med_trajectory[i] != med_trajectory[i-1]]

def _test_unique_trajectory():
    assert unique_trajectory(["etanercept", "baricitinib", "baricitinib", "etanercept"]) == ["etanercept", "baricitinib", "etanercept"]

def map_generic(medication_name, mapping_dict, return_value=False):
    """
    Maps medication name to a generic value based on mapping dict
    Returns None if return_value is False otherwise returns the original value
    """
    keep_original = medication_name
    medication_name = medication_name.lower()
    
    for k in mapping_dict:
        if medication_name in k.lower(): # if query name matches a brand name
            return mapping_dict[k]
        elif medication_name in mapping_dict[k].lower(): # if query name matches a generic name
            return mapping_dict[k]
        elif k.lower() in medication_name: # if brand name is in the query name
            return mapping_dict[k]
        elif mapping_dict[k].lower()+"-" in medication_name: # if generic name + "-" is in the query name (accounting for biosimilars)
            continue
        elif mapping_dict[k].lower() in medication_name: # if generic name is in the query name
            return mapping_dict[k]
        
    if return_value:
        return keep_original
    return None

def split_prompt_test(med_class_name, pt_frac=0.2, random_state=0):
    """
    pt_frac = Fraction of patients to get information from
    """
    ## Load notes
    notes_df = pd.read_parquet(f"./data/{med_class_name}/annotated_medications.parquet.gzip")

    ## Only limit to encounters where there was a medication change
    notes_df = notes_df[notes_df["curr_med_change"]]

    # Split prompt dev/test set
    pts_list = list(notes_df["patientdurablekey"].unique())

    test_pts, prompt_pts = train_test_split(pts_list, test_size=pt_frac, random_state=random_state, shuffle=True)
    #test_pts, valid_pts = train_test_split(test_pts, test_size=0.50, random_state=0, shuffle=True)
    #train_df = notes_df[notes_df["patientdurablekey"].isin(train_pts)]
    prompt_dev_df = notes_df[notes_df["patientdurablekey"].isin(prompt_pts)]
    test_df = notes_df[notes_df["patientdurablekey"].isin(test_pts)]

    #train_df.to_parquet(f"./data/{med_class_name}/gpt4/train.parquet.gzip")
    prompt_dev_df.to_parquet(f"./data/{med_class_name}/gpt4/validation.parquet.gzip")
    test_df.to_parquet(f"./data/{med_class_name}/gpt4/test.parquet.gzip")

def _manual_json_formatting(text):
    """Extract JSON from response. Returns json (dict) if available, else None"""

    # deal with extra spaces and bad characters
    text = re.sub(r'\t', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub("[ ]*,[ ]*",",", text)
    text = re.sub("[ ]*{[ ]*","{", text)
    text = re.sub("[ ]*:[ ]*",":", text)
    text = re.sub("[ ]*}[ ]*","}", text)
    text = text.replace("\x07", "")
    
    try:
        text = text.split("{", 1)[-1]
        text = text[:text.rindex("}")]
        text = text.strip()
        add_curly_brace = text.count("{") - text.count("}")
        text = text+"}"*add_curly_brace
        text = "{"+text+"}"

        # Deal with ' and " values in text
        #if " is not next to a comma or parenthesis, make it a \'
        #},'
        text = re.sub(r"(?<!([,:}{]))\"(?!([,:}{]))", "\'", text)
        text = re.sub(r"(?<!(\",))(?<=([,]))\"(?!([,:}{]))", "\'", text)
        text = re.sub(r"(?<=([,:}{]))\'(?!([,:}{]))", "\"", text)

        return json.loads(text)
    except:
        return text
    
def format_gpt_json_response(response):
    """
    Used with prompt that places values in {"json":str, "format":list<str>}"
    """
    # Get prompt information
    query_values = {}
    query_values["prompt_tokens"] = response.usage.prompt_tokens
    query_values["completion_tokens"] = response.usage.completion_tokens

    # extract text
    text = response.choices[0].message.content
    query_values["response"] = text
    
    if "json" in text:
        text = text.strip('`\n')
        text = text.replace("json", "")
    
    try:
        text = json.loads(text)
    except:
        text = _manual_json_formatting(text)
        
    try:
        query_values.update(text)
        return query_values
    except:
        print("Error converting response to json")
        query_values["json_error"] = text
        return query_values
    
def openai_query(note_keys,
                 note_texts,
                 task,
                 outfile,
                 sys_message=None,
                 functions=None,
                 save_every=15,
                 **api_config):
    '''
    Given a set of notes (with columns "deid_note_key" and "note_text"), query GPT4 with the following prompt template:
    prompt = f'Clinical note: """{note}"""\n{task}'
    '''

    # Query parameters
    if functions is not None:
        kwargs.update({"functions":functions})
        
    # default configs
    default_config = {"model":"gpt-4-turbo-128k",
                        "max_tokens":1024,
                        "frequency_penalty":0,
                        "presence_penalty":0,
                        "temperature":0,
                        "top_p":1,
                     }
    
    default_config.update(api_config)
    api_config = default_config

    # Querying
    response_dict = {}
    curr_ind=0
    print("SYS MESSAGE:", sys_message)
    print("PROMPT:",task)
    
    for note_key, note in zip(note_keys, note_texts):
        # format messages, with system message if appropriate
        prompt = f'Clinical note: """{note}"""\n{task}'
        messages = [{"role": "user","content": prompt}]
        
        if sys_message is not None:
            messages = [{"role": "system", "content": sys_message}] + messages
        
        # get responses
        try:
            response = client.chat.completions.create(messages=messages,  **api_config)
        except:
            time.sleep(5)
            print("time")
            response = client.chat.completions.create(messages=messages,  **api_config)
        
        # format and add metadata
        response_clean = format_gpt_json_response(response)
        response_clean["task"] = task
        response_clean.update(api_config)
        
        if sys_message is not None:
            response_clean["sys_message"] = sys_message
        
        # add to running responses
        response_dict[note_key] = response_clean
        
        # save periodically
        if curr_ind%save_every==0:
            print("Saving up to note key:", note_key)
            responses_df = pd.DataFrame.from_dict(response_dict, orient="index")
            responses_df.to_csv(outfile)

        curr_ind = curr_ind+1

    responses_df = pd.DataFrame.from_dict(response_dict, orient="index")
    responses_df.to_csv(outfile)

def cleanNotes(nlp_df):
    """
    Clean notes by removing extra new lines and "*****" -> ""
    """
    nlp_df["note_text_clean"] = [re.sub(r'\n\s*\n', '\n', s) for s in nlp_df["note_text"]]
    nlp_df["note_text_clean"] = [s.replace("*****", "") for s in nlp_df["note_text_clean"]]
    return nlp_df