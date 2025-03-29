############# IMPORTS #############
### Common imports
# data 
import pandas as pd
import numpy as np

# system 
import os
import re
import datetime

# math and formatting
from scipy.stats import mannwhitneyu
#from umap import UMAP 

# plotting
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mpl_colors

# NLP
from bertopic import BERTopic
import spacy

sns.set_style("white")
sns.set_context("talk")

"""
#import kaleido
#from bioinfokit import analys, visuz
#import pyodbc

# ML
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import auc, plot_roc_curve, classification_report
#import spacy
#import scispacy
#import spacy_transformers
"""

############# PREPROCESSING - QC #############

############# PREPROCESSING - PLOTTING #############

############# MEDICATION SWITCHING #############

def _hex_to_rgba(hex_val, alpha=1):
    hex_val = hex_val.strip("#")
    rgb = [int(hex_val[i:i+2], 16) for i in (0, 2, 4)]
    return f'rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha})'

def _formatSwitch(switch_df, index_dict, color_dict, switch=["1", "2"]):
    """
    Formats medication values in plotly Sankey happy format
    """
    switch_df = switch_df.groupby(switch).count().reset_index()

    switch_df["first_index"] = [index_dict[m] for m in switch_df[switch[0]]]
    switch_df["second_index"] = [index_dict[m] for m in switch_df[switch[1]]]
    switch_df["first_index_label"] = [m for m in switch_df[switch[0]]]
    switch_df["second_index_label"] = [m for m in switch_df[switch[1]]]
    
    # Add color
    switch_df["node_color_hex"] = [color_dict["-".join(m.split("-")[1:])] for m in switch_df[switch[0]]]
    switch_df["node_color"] = [_hex_to_rgba(c, alpha=1) for c in switch_df["node_color_hex"]]
    switch_df["link_color"] = [_hex_to_rgba(c, alpha=0.6) for c in switch_df["node_color_hex"]]
    
    return switch_df[["first_index", "second_index", "node_color", "link_color",
                      "Count", "first_index_label", "second_index_label"]]

def _plotPatientTrajectorySankey(ra_meds_switch, labels, node_colors):
    """
    Actual plotting for Sankey
    """

    # sankey figure
    fig = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(color = "black", width = 0.5),
          label = labels,
          color = node_colors #"grey"
        ),
        link = dict(
          source = ra_meds_switch["first_index"], # indices correspond to labels, eg A1, A2, A1, B1, ...
          target = ra_meds_switch["second_index"],
          value = ra_meds_switch["Count"], 
            label=ra_meds_switch["first_index_label"],
            color = ra_meds_switch["link_color"]),  
        textfont = dict( family = "arial", size = 16))],
                    layout = go.Layout(autosize=False,width=850, height=650))

    # update names
    #TODO: figure out a better way to split multiple indices
    # new_labels = [t.split("-")[1] if "-" in t else t for t in fig.data[0]["node"]["label"]]
    new_labels = ["-".join(t.split("-")[1:]) if "-" in t else t for t in fig.data[0]["node"]["label"]]
    
    for trace in fig.data:
        trace.update(node={"label":new_labels}, visible=True)
            
    # update colors

    fig.update_layout(title_text="Medication switching", font_size=10)
    return fig

def plotPatientTrajectorySankey(ehr_df, time_col = 'startdatekeyvalue', patient_col= "patientdurablekey", 
                          values_col = "MedicationClass", switches=["1","2","3"], palette="pastel", save_fig=None):
    """
    Organize data for Plotly Sankey diagram plotting
    Plots number of patients in discrete values
    
    Params:
        time_col (str): column to sort values by
        patient_col (str): column containing patients to group by
        values_col (str): column containing values (eg. medication <str> or medication trajectory <list>) for Sankey labels
        switches (list<str>): Switches to plot
        palette (str, sns.palette): Seaborn palette
        
    Returns:
        Tuple<go.Figure, pd.DataFrame>

    """
    
    if type(ehr_df.iloc[0][values_col]) is str:
        # extract values
        ehr_df["groupedLabels"] = ehr_df[patient_col].map(extractInstance(ehr_df, patient_col=patient_col, n=None, 
                                                                      time_col=time_col, values_col=values_col))
    else:
        ehr_df["groupedLabels"] = ehr_df[values_col]

    # collapse values by patient
    ehr_df["nGroupedLabels"] = [m[:(int(switches[-1]))] if len(m)>(int(switches[-1])-1)  else m for m in ehr_df["groupedLabels"]]
    ehr_df = ehr_df.groupby(patient_col).first()
    
    # create labels and plot values
    labels = []
    for s in switches:
        ehr_df[s] = [(s)+"-"+m[int(s)-1] if len(m)>(int(s)-1) else s+"-No switch" for m in ehr_df["groupedLabels"]]
        labels.extend(list(ehr_df[s]))

    labels = list(set(labels))
    index_dict = dict(zip(labels, range(len(labels))))
    node_colors = [palette["-".join(l.split("-")[1:])] for l in labels]
    
    # format switch for sankey
    ra_meds_switch = pd.DataFrame()
    ehr_df["Count"] = 1
    for i in range(len(switches)-1):
        curr_switch = _formatSwitch(ehr_df,
                                    index_dict,
                                    palette,
                                    switch=[switches[i],
                                            switches[i+1]])
        ra_meds_switch = pd.concat([ra_meds_switch, curr_switch])
        
    fig = _plotPatientTrajectorySankey(ra_meds_switch, labels, node_colors) 
    
    # save
    if save_fig:
        fig.write_image(save_fig, format=save_fig.split(".")[-1])
        
    return fig, ra_meds_switch
