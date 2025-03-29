import pandas as pd
import numpy as np
import tiktoken

from utils.dask_cluster import load_register_table, load_cluster
from utils.medswitch import unique_trajectory, fill_prev_next_meds, map_generic, get_first_or_last_values
from sklearn.model_selection import train_test_split
from utils.benchmark.metrics import classification_metrics, substring_search_metrics

def getMedications(ncdmard_mapping, output="./data/ncdmard/raw"):
    """
    Retrieve all relevant medication orders (and medications given as procedures) and save raw data
    Procedures are restricted to those with procedurecategory starting with "PR " for medication procedures
    """
    # Get all medication names
    med_list = list(ncdmard_mapping.keys()) + list(ncdmard_mapping.values())
    med_list = list(set(med_list))

    # load tables
    orders_table = load_register_table("DEID_CDW", "medicationorderfact")
    proc_table = load_register_table("DEID_CDW", "procedureeventfact")
    
    # Get all medication orders
    med_orders = orders_table[orders_table["medicationname"].str.contains("|".join(med_list), case=False)|
                             orders_table["medicationgenericname"].str.contains("|".join(med_list), case=False)].compute()
    print("all relevant medications:", med_orders.shape)
    med_orders.to_parquet(f"{output}/ncdmard_med_orders.parquet.gzip")
    
    # get all procedures
    med_proc = proc_table[proc_table["procedurename"].str.contains("|".join(med_list), case=False)|
                             proc_table["procedurename"].str.contains("|".join(med_list), case=False)].compute()
    med_proc = med_proc[med_proc["procedurecategory"].str.startswith("PR ")]
    print("all relevant procedures:", med_proc.shape)
    med_proc.to_parquet(f"{output}/ncdmard_procedures.parquet.gzip")

def getDiagnosis(diag_codes, exclude_codes, output="./data/ncdmard/raw"):
    """
    Get diagnosis for all patients who have any diagnosis provided and no exclusion codes
    """
    # Get relevant diagnosis terms
    diag_term_table = load_register_table("DEID_CDW", "diagnosisterminologydim")
    ra_diag_terms = diag_term_table[diag_term_table["value"].str.contains(diag_codes, na=False)]
    ra_diag_terms = ra_diag_terms[["diagnosisname", "type", "value"]]
    ra_diag_terms.columns = ["diagnosisname", "ontology", "ontology_value"]
    ra_diag_terms = ra_diag_terms.compute()
    
    # get relevant diagnosis (removed dask merging to avoid issues)
    diagnosis_table = load_register_table("DEID_CDW", "diagnosiseventfact")
    ra_table = diagnosis_table[diagnosis_table["diagnosisname"].isin(list(ra_diag_terms["diagnosisname"].unique()))]
    ra_table = ra_table.compute()
    
    ra_table = ra_table.merge(ra_diag_terms, how="inner", left_on="diagnosisname", right_on="diagnosisname")
    print("Number of patients with all diagnosis codes:", ra_table.shape)
    
    # Remove patients with exclusion codes
    exclude_pts = ra_table[ra_table["ontology_value"].str.contains(exclude_codes)]
    exclude_pts = list(exclude_pts["patientdurablekey"].unique())
    print("Number of patients with exclusion codes:", len(exclude_pts))
    ra_table = ra_table[~ra_table["patientdurablekey"].isin(exclude_pts)]
    
    # save
    ra_table.to_parquet(f"{output}/ncdmard_diagnosis.parquet.gzip")

def getDemographics(filepath="./data/ncdmard/raw"):
    """
    Gets RA patient demographic information and merge information needed for filtering (medications, encounter dates, age, etc)
    """
    # Load diagnosis and medications/procedures
    med_proc = pd.read_parquet(f"{filepath}/ncdmard_procedures.parquet.gzip")
    med_orders = pd.read_parquet(f"{filepath}/ncdmard_med_orders.parquet.gzip")

    # Merge medications and procedure tables (use ordered date if start date is not available)
    med_orders["startdatekeyvalue"] = [o if s is None else s for o,s in zip(med_orders["ordereddatekeyvalue"], med_orders["startdatekeyvalue"])]
    med_orders = med_orders[["patientdurablekey", "medicationorderkey", "encounterkey","medicationname", "medicationgenericname", 
                             "startdatekeyvalue", "enddatekeyvalue", "associateddiagnosiscombokey"]]

    med_proc = med_proc[["patientdurablekey", "procedureeventkey", "encounterkey", "procedurename", 
             "procedurestartdatekeyvalue", "procedureenddatekeyvalue"]]
    med_proc.columns = ["patientdurablekey", "procedureeventkey", "encounterkey", "medicationname", 
             "startdatekeyvalue", "enddatekeyvalue"]
    all_meds = pd.concat([med_orders,med_proc])
    all_meds = all_meds.dropna(subset="startdatekeyvalue") # removes 1 value
    all_meds = all_meds.sort_values("startdatekeyvalue")
    all_meds.to_parquet(f"{filepath}/ncdmards_all_meds.parquet.gzip")
    print("# of DMARD prescriptions", all_meds.shape)
    print("# of unique patients with DMARD prescriptions: ", all_meds["patientdurablekey"].nunique())
    
    # Load patient demographics table and filter to patients with at least 1 relevant drug order
    pts_df = load_register_table("DEID_CDW", "patdurabledim")
    ra_pts = list(all_meds["patientdurablekey"].unique())
    pts_df = pts_df[pts_df["iscurrent"]==1]
    pts_df = pts_df[pts_df["isvalid"]==1]
    pts_df = pts_df[pts_df["patientdurablekey"].isin(ra_pts)]
    pts_df = pts_df.compute()

    print("# patients with demographic information:", pts_df["patientdurablekey"].nunique())

    # Add first medication information to demographics
    sort_meds = all_meds[["medicationname", "medicationgenericname", "startdatekeyvalue", "patientdurablekey"]]
    first_ncdmard = get_first_or_last_values(sort_meds, datetime_col="startdatekeyvalue",
                                             get_first=True, groupby="patientdurablekey",
                                             prefix = "first_ncdmard_")

    pts_df = pts_df.merge(first_ncdmard, left_on="patientdurablekey",
                          right_index=True, how="left")
    
    # Add last medication information to demographics
    last_ncdmard = get_first_or_last_values(sort_meds, datetime_col="startdatekeyvalue",
                                             get_first=False, groupby="patientdurablekey",
                                             prefix = "last_ncdmard_")
    pts_df = pts_df.merge(last_ncdmard, left_on="patientdurablekey",
                          right_index=True, how="left")

    # Get last encounter for each patient
    ra_pts = list(pts_df["patientdurablekey"].unique())
    encounter_table = load_register_table("DEID_CDW", "encounterfact")
    encounter_table = encounter_table[encounter_table["patientdurablekey"].isin(ra_pts)]
    encounter_table = encounter_table.dropna(subset=["datekeyvalue"])
    ra_encounters = encounter_table.compute()
    ra_encounters = ra_encounters.sort_values("datekeyvalue", ascending=True)
    ra_encounters = ra_encounters.groupby("patientdurablekey").last()
    
    # Add last encounter values to patient demographics
    ra_encounters = ra_encounters[["encounterkey", "datekeyvalue"]]
    ra_encounters.columns = ["last_encounterkey", "last_encounter_datekeyvalue"]
    pts_df = pts_df.merge(ra_encounters, left_on="patientdurablekey",
                          right_index=True, how="left")

    print(pts_df.shape)
    pts_df.to_parquet(f"{filepath}/ncdmards_all_pts.parquet.gzip")
    
def addNotes(filepath="./data/ncdmard/raw", meds_min_months_fu=6):
    """
    Filter patients and add clinical notes
    """
    # load patient & medication values
    pts_df = pd.read_parquet(f"{filepath}/ncdmards_all_pts.parquet.gzip")
    meds_df = pd.read_parquet(f"{filepath}/ncdmards_all_meds.parquet.gzip")

    # Add in metadata on absolute values for patients
    pts_df["last_ncdmard_to_last_encounter_days"] = [None if pd.isna(d) else d.days for d in (pts_df["last_encounter_datekeyvalue"] - pts_df["last_ncdmard_startdatekeyvalue"])]
    pts_df["treated_with_ncdmard"] = ~pts_df["first_ncdmard_startdatekeyvalue"].isna()
    
    # Filter medications (and patients) to those with at least 3 months of follow up
    meds_df["last_encounter_datekeyvalue"] = meds_df["patientdurablekey"].map(dict(zip(pts_df["patientdurablekey"], pts_df["last_encounter_datekeyvalue"])))
    meds_df["start_med_to_last_encounter_days"] = [None if pd.isna(d) else d.days for d in (meds_df["last_encounter_datekeyvalue"] - 
                                                                                              meds_df["startdatekeyvalue"])]
    
    meds_df["has_follow_up"] = meds_df["start_med_to_last_encounter_days"]>=(30*meds_min_months_fu)
    print(f"medications with final encounter >={meds_min_months_fu} months start:", meds_df["has_follow_up"].value_counts())

    # Save filtered pts_df
    pts_df.to_parquet(f"{filepath}/ra_cohort_pts.parquet.gzip")
    
    # Filter medications to relevant patients
    meds_df = meds_df[meds_df["patientdurablekey"].isin(pts_df["patientdurablekey"].unique())]
    print("# of medications for new filtered patients (with demographics):", meds_df.shape)

    # Get clinical notes written by specialists associated with medication encounters
    notes_rdd = load_register_table("DEID_CDW", "note_text")
    notes_meta_rdd = load_register_table("DEID_CDW", "note_metadata")

    med_encounters = list(meds_df["encounterkey"])
    med_notes_meta = notes_meta_rdd[notes_meta_rdd["encounterkey"].isin(med_encounters)]
    med_notes_meta = med_notes_meta.compute()
    #med_notes_meta = med_notes_meta.merge(notes_rdd, left_on="deid_note_key", right_on="deid_note_key", how="inner")
    
    # workaround for merging
    notes_rdd = notes_rdd[notes_rdd["deid_note_key"].isin(list(med_notes_meta["deid_note_key"].unique()))]
    notes_rdd = notes_rdd.compute()
    med_notes_meta = med_notes_meta.merge(notes_rdd, left_on="deid_note_key", right_on="deid_note_key", how="inner")

    # Add clinical notes to medication tables
    med_notes_meta = med_notes_meta[["note_text", "prov_specialty", "encounter_type", "note_type", 
                    "encounterkey", "enc_dept_name", "enc_dept_specialty", "deid_service_date", "deid_note_key"]]
    med_notes_meta.columns = ["note_"+s if not s.startswith("note") else s for s in med_notes_meta.columns]

    meds_df = meds_df.merge(med_notes_meta, left_on="encounterkey", right_on="note_encounterkey", how="inner")
    print("# of clinical notes associated with medication orders:", meds_df.shape)
    print("# of patients with relevant medications & notes:", meds_df["patientdurablekey"].nunique())

    # Save medications associated with annotated notes
    meds_df.to_parquet(f"{filepath}/ncdmard_meds_with_notes.parquet.gzip")
    

def finalWeakAnnotations(generic_dict, filepath="./data/ncdmard/raw"):
    """
    Clean up values and add trajectory values
    """
    # Get medications and notes
    med_notes_df = pd.read_parquet(f"{filepath}/ncdmard_meds_with_notes.parquet.gzip")

    # Get patient dataframe
    pts_df = pd.read_parquet(f"{filepath}/ra_cohort_pts.parquet.gzip")

    # Map medication names to generic values
    med_notes_df["mapped_med_generic"] = [m if g=="*Unspecified" 
                                          else m if g is None
                                          else m+g for m,g in zip(med_notes_df["medicationname"],
                                                                med_notes_df["medicationgenericname"])]

    med_notes_df["mapped_med_generic_clean"] = med_notes_df["mapped_med_generic"].apply(map_generic,mapping_dict=generic_dict)

    # drop duplicates medications with the same start date
    # And then group by encounterkey or medication name + date
    print("Original number of medications with clinical notes:", med_notes_df.shape)
    print("Unique number of medications with clinical notes:", med_notes_df["medicationorderkey"].nunique())
    med_notes_df = med_notes_df.sort_values("startdatekeyvalue")
    med_notes_df = med_notes_df.drop_duplicates(subset=["patientdurablekey", "mapped_med_generic_clean", "encounterkey", "startdatekeyvalue"], keep="first")
    print("Drop duplicate medications with same class and same start date:",med_notes_df.shape)
    print("# patients remaining:", med_notes_df["patientdurablekey"].nunique())

    # Drop encounters that have multiple prescriptions (only keep last medication)
    med_notes_df = med_notes_df.groupby(["patientdurablekey", "encounterkey"]).last().reset_index(drop=False)
    
    print("Keep only the last medication for each encounter:",med_notes_df.shape)
    print("# patients remaining:", med_notes_df["patientdurablekey"].nunique())

    # Create previous and next columns
    med_notes_df = fill_prev_next_meds(med_notes_df)

    # Save annotated medications
    med_notes_df.to_parquet("./data/ncdmard/annotated_medications.parquet.gzip")

    # Add medication trajectory for each patient
    final_med_notes_df = med_notes_df[["patientdurablekey", "medicationname", "encounterkey",
                                       "medicationgenericname",  "mapped_med_generic_clean", 
                                       "startdatekeyvalue", "enddatekeyvalue", "note_deid_note_key"]]

    for col in final_med_notes_df.columns:
        med_trajectory = final_med_notes_df.groupby("patientdurablekey", sort=False)[col].apply(list)
        pts_df["final_"+col] = pts_df["patientdurablekey"].map(med_trajectory)

    # Add in unique medication trajectory and labels for patient switching
    pts_df["final_unique_med_trajectory"] = [None if type(t)!=list else unique_trajectory(t) for t in pts_df["final_mapped_med_generic_clean"]]

    pt_final_fu = med_notes_df.sort_values("startdatekeyvalue").groupby("patientdurablekey")["has_follow_up"].last()

    pts_df["med_switching_label"] = ["No TNFi with notes" if u is None
                                         else "TNFi switch" if len(u)>1
                                         else "No switch" for u in  pts_df["final_unique_med_trajectory"]]
    pts_df["has_follow_up"] = pts_df["patientdurablekey"].map(pt_final_fu)
    pts_df["med_switching_label"]  = [l if l!="No TNFi"
                                              else l + " (Lost to follow-up)" if not f
                                              else l for l,f in zip(pts_df["med_switching_label"], pts_df["has_follow_up"])]
    pts_df["med_switching_label"]  = [l if "No TNFi" in l
                                              else l + " (No final follow-up)" if not f
                                              else l for l,f in zip(pts_df["med_switching_label"], pts_df["has_follow_up"])]

    print("Distribution of treatment switching:")
    print(med_notes_df["curr_med_change"].value_counts())

    # Save annotated patient data
    print()
    print("Distribution of treatment switching (Patients):")
    print(pts_df["med_switching_label"].value_counts())
    pts_df.to_parquet("./data/ncdmard/annotated_pt_demographics.parquet.gzip")


def evaluate_ncdmard(prompt_dev_df, 
                      med_mapping,
                      date,
                      med_class_name="ncdmard",
                      engine="gpt4",
                     average="micro"):
    """Full eval loop for prompt development dataset"""
    eval_dfs = {"class":pd.DataFrame(), "text":pd.DataFrame(), "pred_values":pd.DataFrame()}

    #function_config= None # Currently not implemented 
    for sys_config in [""]: #, 
        for task_config in ["all-values-provided", "default-task", "drugs-provided", "reasons-provided"]: # "function", "manual-function-no-type",
            if task_config in ["manual-function-no-type"] and sys_config in ["specialist", "crc",]:
                continue
            
            outfile = f"./data/{med_class_name}/gpt4/prompt_dev/{date}_{engine}_{task_config}_prompt_dev.csv"
            
            gpt4_df = pd.read_csv(outfile, index_col=0)
            gpt4_df = gpt4_df.loc[list(prompt_dev_df["note_deid_note_key"])]

            # Classification values
            class_metrics = {}
            pred_values = {}
            for pred_col, label_col in [("new_ncDMARD","mapped_med_generic_clean"), ("last_ncDMARD","prev_medication")]:
                preds = list(gpt4_df[pred_col])
                
                preds = [None if type(p)!=str else map_generic(p, med_mapping["generic_mapping"]) for p in preds]
                preds = ["" if p is None else p for p in preds]
                labels = list(prompt_dev_df[label_col])
                
                class_metrics[label_col] = classification_metrics(preds=preds, labels=labels, average=average)
                pred_values[label_col+"_preds"] = preds
                pred_values[label_col+"_labels"] = labels

            # Free text values
            text_metrics={}
            if task_config=="sbs":
                labels = list(prompt_dev_df["note_text"])
                note_ids = list(prompt_dev_df["note_deid_note_key"])
                for pred_col in ["new_ncDMARD_sentence", "last_ncDMARD_sentence", "reason_last_ncDMARD_stopped_sentence"]:
                    # Get predictions
                    preds = list(gpt4_df[pred_col])
                    preds = ["" if type(p)!=str 
                             else "" if p=="NA"
                             else p for p in preds]

                    # Get substring scores
                    substring_scores = substring_search_metrics(preds, labels, ignore_empty_preds=True)
                    substring_scores = [s/100 for s in substring_scores]
                    text_metrics[pred_col] = {"mean":np.mean(substring_scores), 
                                            "std":np.std(substring_scores), 
                                            "median":np.median(substring_scores)}
                    
                    # Store predictions and labels
                    pred_values[pred_col+"_preds"] = preds
                    pred_values[pred_col+"_labels"] = note_ids
                    pred_values[pred_col+"_score"] = substring_scores

            # Update full dataframe and store current values
            # metric_set = ["class", "text", "pred"]
            for metric_set_name, curr_metrics in zip(eval_dfs, [class_metrics, text_metrics, pred_values]):
                all_class_df = eval_dfs[metric_set_name]
                curr_class_df = pd.DataFrame.from_dict(curr_metrics, orient="index")

                curr_class_df["sys_config"] = sys_config
                curr_class_df["task_config"] = task_config
                all_class_df = pd.concat([all_class_df, curr_class_df])
                eval_dfs[metric_set_name] = all_class_df

                    
    eval_dfs["text"].to_csv(f"./data/ncdmard/gpt4/eval/prompt_dev_text_metrics_{average}.csv")
    eval_dfs["class"].to_csv(f"./data/ncdmard/gpt4/eval/prompt_dev_classification_metrics_{average}.csv")
    eval_dfs["pred_values"].to_csv("./data/ncdmard/gpt4/eval/prompt_dev_evaluated_preds.csv")
    
