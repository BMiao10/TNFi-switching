import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import evaluate

def get_attention_mask(input_ids):
    """Returns attention mask values for input_ids list or np.array. Not sure why I need this."""
    if type(input_ids)==list:
        return [[1 if e!=0 else 0 for e in encoding] for encoding in input_ids]
    else:
        return [np.where(encoding==0,0,1).tolist() for encoding in input_ids]
    
def acc_ci(accuracy, n, z=1.96):
    """
    Confidence interval accuracy
    
    Parameters:
        accuracy (float): accuracy score
        n (int): number of values
        z (float): z score values corresponding to confience of interest

    Z score values:
        1.64 (90%)
        1.96 (95%)
        2.33 (98%)
        2.58 (99%)
    
    Returns: 
        float
    """
    
    return z * np.sqrt( (accuracy * (1 - accuracy)) / n)

def classification_metrics(preds, labels, average="micro"):
    """
    General classification metrics (acc, F1, precision, recall)
    Default "micro" averaging takes imbalance classes into account
    See sklearn metrics for more information: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
    """
    return {"acc": accuracy_score(y_pred=preds, y_true=labels, normalize=False),
            "acc_norm": accuracy_score(y_pred=preds, y_true=labels, normalize=True), 
            f"{average}_F1": f1_score(y_pred=preds, y_true=labels, average=average, labels=np.unique(labels)),
            f"{average}_precision": precision_score(y_pred=preds, y_true=labels, average=average),
            f"{average}_recall":recall_score(y_pred=preds, y_true=labels, average=average)}


def substring_search_metrics(preds, references, ignore_empty_preds=False):
    """
    Returns score of how closely the each prediction matches a substring in the corresponding reference
    Parameters:
        preds (list<str>): predictions
        references (list<str>): labels
        ignore_empty_preds (bool): if True, only calculates scores for non-empty prediction strings

    Returns: list<float>
    """ 
    if ignore_empty_preds:
        return [fuzz.partial_ratio(p, r) for p, r in zip(preds, references) if len(p)>0]
    
    return [fuzz.partial_ratio(p, r) for p, r in zip(preds, references)]

def compute_bleu_score(preds, references, tokenizer=None, max_order=1, **kwargs):

    """
    Madhumita Sushil
    :param preds: list of all predictions
    :param references: list of list of all references for all predictions.
    :return: results
    """
    
    bleu = evaluate.load("bleu")
    if tokenizer is not None:
        results = bleu.compute(predictions=preds, references=references, smooth=False,
                               tokenizer=tokenizer.tokenize, max_order=max_order, **kwargs)
    else:
        results = bleu.compute(predictions=preds, references=references, smooth=False, max_order=max_order, **kwargs)
    return results

def compute_rouge_score(preds, references, rouge_types=None, tokenizer=None):
    """
    Madhumita Sushil
    :param preds: list of all predictions
    :param references: list of list of all references for all predictions.
    :param rouge_types:
    :return: results
    """
    rouge = evaluate.load("rouge")
    if tokenizer is not None:
        results = rouge.compute(predictions=preds, references=references, rouge_types=rouge_types,
                                tokenizer=tokenizer.tokenize)
    else:
        results = rouge.compute(predictions=preds, references=references, rouge_types=rouge_types,
                                use_stemmer=True)
    return results

def set_em_accuracy(pred, label, z=1.96):
    """
    Exact match set (difference) accuracy
    
    Parameters:
        pred (list<list>): list of predicted values 
        label (list<list>): list of ground truth values to compare to
    
    """
    correct = [1 if len(set(annot).difference(set(p)))==0  
               else 0 for p, annot in zip(pred, label)]
    accuracy = sum(correct)/len(label)
    interval = acc_ci(accuracy, n=len(label), z=z)
    print("Accuracy: %.3f;"%accuracy, "CI: %.3f;"%interval, "n: %s"%len(label), )
    return correct
    
def set_f1_scores(preds, references, average="macro", **kwargs):
    """
    Converts list<list> values into F1 compatible values
    Compares values joined as str with ","
    """
    preds = ["None" if type(x)!=list else "None" if len(x)==0 else ",".join(x) for x in preds]
    references = ["None" if type(x)!=list else "None" if len(x)==0 else ",".join(x) for x in references]
    vals_dict = {val:i for i, val in enumerate(np.unique(preds+references))}
    
    preds = [vals_dict[x] for x in preds]
    references = [vals_dict[x] for x in references]
    
    return f1_score(y_pred=preds, y_true=references, average=average, labels=np.unique(references), **kwargs)
    
    