import os
import sys
import json
from tqdm import tqdm

import fire
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from torch.utils.data import Dataset, DataLoader

def load_model(model_name_or_path,
               device_map="auto",
               local_files_only=True,
               load_in_8bit=True,
               **kwargs):
    if not torch.cuda.is_available():
        print("No cuda found")
        load_in_8bit = False
        
    # set up quantization
    quantization_config = BitsAndBytesConfig(load_in_8bit=load_in_8bit)
    
    # load model
    print(f"Loading model {model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                 local_files_only=local_files_only,
                                                 device_map=device_map,
                                                 #torch_dtype=torch.bfloat16,
                                                 quantization_config=quantization_config,
                                                 **kwargs
                                                 )
    print("Completed model loading")
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
        
    return model

def load_tokenizer(model_name_or_path,
                   local_files_only=True,
                   padding_side='left',
                   **kwargs):
    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                              padding_side=padding_side,
                                              local_files_only=local_files_only,
                                              **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Completed loading tokenizer")
    return tokenizer

def load_base_configs(cfg_options):
    default_model_cfgs = {"device_map":"auto",
                          "local_files_only":True,
                          "load_in_8bit":True}
    default_tok_cfgs = {"padding_size":"left",
                        "local_files_only":True}
    if "model_config" in cfg_options:
        default_model_cfgs.update(cfg_options["model_config"])
    if "tokenizer_config" in cfg_options:
        default_tok_cfgs.update(cfg_options["tokenizer_config"])
        
    return default_model_cfgs, default_tok_cfgs

def load_inference_configs(cfg_options):
    default_inf_cfgs = {"max_new_tokens":250,
                          "top_p":1,
                          "do_sample":False}
    if "generation_config" in cfg_options:
        default_inf_cfgs.update(cfg_options["generation_config"])
        
    return default_inf_cfgs


def format_hf_chat_template(tokenizer,
                            batch_notes,
                            task,
                            sys_message=None,
                           truncate_note_to=6000):
    """
    Generate a single tokenized prompt from list of messages for chat-tuned models.

    Messages should be a list of messages, each of which should be in the following format:
    messages = [
            {"role": "system", "content": "Provide the system message content here."},
            {"role": "user", "content": "Provide the user content here."},
            {"role": "assistant", "content": "Provide the assistant content here."},
        ]
    formatted_messages = [messages]
    
    tokenizer: Instance of Huggingface tokenizer
    model: Instance of Huggingface model
    gen_config: dictionary of configuration parameters and values for generating model response.
    """    
    formatted_messages = list()

    for note in batch_notes:
        # Truncate note if necessary (slower but better than truncating whole prompt)
        if truncate_note_to is not None:
            trunc_note = tokenizer(note, padding=False)
            if len(trunc_note["input_ids"]) > truncate_note_to:
                trunc_note = tokenizer.batch_decode(sequences=trunc_note["input_ids"][:truncate_note_to],
                                                    skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=True)
                note = " ".join(trunc_note)
                print("Truncated note to length", len(trunc_note))
        
        # Format into "message" format
        if sys_message is None:
            message = [
                {
                    'role': 'user',
                    'content': f"Clinical note: \"\"\"{note}\"\"\"\n{task}" 
                }
            ]
        else:
            message = [
                        {
                            'role': 'system',
                            'content': sys_message
                         },

                        {
                            'role': 'user',
                            'content': f"Clinical note: \"\"\"{note}\"\"\"\n{task}"
                        }
                    ]
        
        # default chat template if none available
        if tokenizer.chat_template is None:
            formatted_messages.append(str(message))
        else: # apply chat template
            format_msg = tokenizer.apply_chat_template(message,
                                                       tokenize=False,
                                                       add_generation_prompt=True)
            formatted_messages.append(format_msg)

    tokenized_messages = tokenizer(formatted_messages,
                                   return_token_type_ids=False,  # token_type_ids are not needed for generation?
                                   return_tensors="pt",
                                   padding=True)

    return tokenized_messages, formatted_messages

# Create dataset class
class NotesDataset(Dataset):               
    def __init__(self, data_fpath):  
        # Read in data
        if "csv" in data_fpath:
            pts_df = pd.read_csv(data_fpath)
        else:
            pts_df = pd.read_parquet(data_fpath)

        print("Remaining notes", pts_df.shape)
        
        # Get text values
        if "deid_note_key" not in pts_df.columns:
            if "note_deid_note_key" in pts_df.columns:
                print("Using note_deid_note_key as note key")
                pts_df["deid_note_key"] = pts_df["note_deid_note_key"]
            else:
                print("Using index as note key")
                pts_df["deid_note_key"] = pts_df.index
            
        self.text = list(pts_df["note_text"])
        self.idxs = list(pts_df["deid_note_key"])

    def __len__(self):                  
        return len(self.idxs)

    def __getitem__(self, i):
        return self.text[i], self.idxs[i] # tokens, index 

def batch_format_values(batch_decoded,
                        batch_idx):
    batch_dict = {}
    
    for text, idx in zip(batch_decoded, batch_idx):
        # Parse data
        response_dict = {"full_text_response":text}
        text = text.split("{", 1)[-1]
        if "}" in text:
            text = text[:text.rindex("}")]
        text = text.strip(" {}")
        add_curly_brace = text.count("{") - text.count("}")
        text = text+"}"*add_curly_brace
        text = "{"+text+"}"

        # Additional parsing
        data = text.replace(", ", ",")
        data = data.replace("False", "false")
        data = data.replace("True", "false")
        data = data.replace("\_", "_")

        try:
            data_dict = json.loads(data)
            response_dict.update(data_dict)
        except:
            print("parsing error")
            response_dict.update({"error":str(data)})
        
        batch_dict[idx] = response_dict
             
    return batch_dict

def run_open_source_baseline(model_config_fpath,
                             data_fpath, 
                             out_dir,
                             out_file_name,
                             task,
                             sys_message=None,
                             batch_size=16,
                             truncate_note_to=6000
                            ):
    '''
    Open source model inference over ClinicalNoteDataset with specific task and system message
    Clinical note: """{note}"""  task
    
    Parameters:
        model_config_fpath (str): 
        data_fpath (str): CSV or parquet file containing columns "deid_note_key" and "note_text" or equivalent 
        out_dir (str): file directory to write outputs to
        out_file_name (str): name of file to write outputs to 
        task (str): request for information to retrieve using LLM
        sys_message (str, default=None): system message to use
        batch_size (int, default=16): batch size to run
        truncate_note_to (int, None, default=6000): length to truncate note to if needed
    '''
    # load model configs
    with open(model_config_fpath) as cfg_file:
        cfg_options = json.load(cfg_file)

    if 'olmo' in cfg_options["model_name"].lower():
        import hf_olmo # Not sure how to improve dynamic loading here. TODO: brainstorm

    model_name_or_path = cfg_options["model_name"]
    if "model_path" in cfg_options:
        model_name_or_path = os.path.join(cfg_options["model_path"], cfg_options["model_name"])

    # Set default inference options and update if neeeded
    model_cfgs, tokenizer_cfgs = load_base_configs(cfg_options)
    gen_kwargs = load_inference_configs(cfg_options)

    # Loading model and tokenizer
    model = load_model(model_name_or_path, 
                       **model_cfgs)
    tokenizer = load_tokenizer(model_name_or_path, 
                               **tokenizer_cfgs)
    
    # Load data and convert to dataset
    # TODO: make the tokenization in the collate function
    notes_dataset = NotesDataset(data_fpath)
    dataloader = DataLoader(notes_dataset,
                            batch_size=batch_size,
                            shuffle=False)

    # Set some options
    only_new_text = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # handle paths
    if not os.path.exists(os.path.realpath(out_dir)):
        os.makedirs(os.path.realpath(out_dir))
    
    outfile = os.path.join(out_dir, out_file_name)
    print("Outputs will be written to", outfile)

    # Evaluation in batches
    model.eval()
    all_response_df = pd.DataFrame()
    for batch_notes, batch_idx in tqdm(dataloader):
        # Tokenize
        tokenized_msgs, prompts = format_hf_chat_template(tokenizer,
                                                 batch_notes,
                                                 sys_message=sys_message,
                                                 task=task,
                                                truncate_note_to=truncate_note_to)
        tokenized_msgs = tokenized_msgs.to(device)

        # Generate and extract output
        try:
            outputs = model.generate(**tokenized_msgs,
                                     pad_token_id=tokenizer.pad_token_id,
                                     eos_token_id=tokenizer.eos_token_id,
                                     **gen_kwargs)
        except:
            print(f"Error running: {batch_idx}")
            continue
            
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        if only_new_text:
            # Using only the first decoded tokens since we only pass one instance.
            batch_decoded = tokenizer.batch_decode(sequences=tokenized_msgs['input_ids'], skip_special_tokens=True)
            prompt_lens = [len(prompt) for prompt in batch_decoded]
            decoded = [cur_response[cur_prompt_len:] for cur_response, cur_prompt_len in zip(decoded, prompt_lens)]

        # Format and add task info
        batch_dict = batch_format_values(decoded, batch_idx)
        batch_df = pd.DataFrame.from_dict(batch_dict, orient="index")

        batch_df["sys_message"] = sys_message
        batch_df["task"] = task
        batch_df["full_prompts"] = prompts
        
        # Save
        all_response_df = pd.concat([all_response_df, batch_df])
        all_response_df.to_csv(outfile)
            
if __name__ == '__main__':
    fire.Fire(run_open_source_baseline)