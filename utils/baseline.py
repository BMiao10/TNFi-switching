### IMPORTS
import numpy as np
import pandas as pd

#import torch 
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

import optuna
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import AdamW, get_linear_schedule_with_warmup
from simpletransformers.classification import ClassificationModel, ClassificationArgs
'''
'''
from utils import medswitch
from utils.benchmark import preprocess
print("done importing")

### PARAMS
def ml_baseline(target_col = "prev_medication",
                date = "2023-11-13",
                model_name = "rf",
                features = "bow",
                med_class_name = "contraceptives",
               curr_meds_only=True):
    print(f"./data/{med_class_name}/baseline/{date}_kfold_{model_name}_{target_col}_{features}.csv")
        
    ## Load annotations
    annot_med = pd.read_parquet(f"./data/{med_class_name}/annotated_medications.parquet.gzip")
    if curr_meds_only:
        annot_med = annot_med[annot_med["curr_med_change"]]
    
    annot_med = medswitch.cleanNotes(annot_med)

    # Shuffle values
    benchmark_df = annot_med[["patientdurablekey",
                              "prev_medication", 
                              "mapped_med_generic_clean",
                              "next_medication",
                              "next_med_change",
                             "note_text_clean"]].sample(frac=1)

    # Create feature dataframes
    bow, tfidf, y = preprocess.split_input_label(benchmark_df, 
                                                 text_col="note_text_clean", 
                                                 target=target_col)

    groups = list(benchmark_df["patientdurablekey"])
    if features=="bow":
        # Get KFold splits
        kfolds = preprocess.kfold_split(X=bow.values,
                                     y=y[0], 
                                     pids=groups, 
                                     n_splits=5, 
                                     **{"train_size":0.875})
    elif features=="tfidf":
        kfolds = preprocess.kfold_split(X=tfidf.values,
                                                 y=y[0], 
                                                 pids=groups, 
                                                 n_splits=5, 
                                                 **{"train_size":0.875})

    # Grid search for hyper parameters
    if model_name =="rf":
        param_grid = {'n_estimators': [50, 100, 250, 500],
                 'max_depth': [20, 50, 100]}
    elif model_name =="logreg":
        param_grid = {'C': [0.01, 0.1, 1, 10, 100, 1000]}

    # Testing on best model
    # With subsampling for training data
    full_report_df = pd.DataFrame()

    full_predict = []
    full_y = []

    for i,kfold in enumerate(kfolds): #kfolds_tfidf
        print("Kfold "+str(i))

        train_X, valid_X, test_X = kfold["X_vals"]
        train_y, valid_y, test_y = kfold["y_vals"]

        if model_name =="logreg":
            clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', class_weight="balanced", max_iter=250)
        elif model_name =="rf":
            clf = RandomForestClassifier(random_state=0)

        # Grid search for hyperparameters
        ps = PredefinedSplit(test_fold=[-1]*len(train_X) + [0]*len(valid_X))
        grid_search = GridSearchCV(clf, param_grid, cv=ps, scoring='f1_macro')

        train_grid_X = np.concatenate((train_X, valid_X), axis=0)
        train_grid_y = np.concatenate((train_y, valid_y), axis=0)

        grid_search.fit(train_grid_X, train_grid_y);
        params = grid_search.best_params_
        print(params)

        if model_name =="logreg":
            clf = LogisticRegression(multi_class='multinomial', solver='lbfgs',
                                     class_weight="balanced", max_iter=250, **params)
        elif model_name =="rf":
            clf = RandomForestClassifier(random_state=0, **params)

        # If needed, subsample training dataset to simulate few-shot experiments (100%, 50%, 25%, 10%, 5%, 1%) 
        # based on https://arxiv.org/pdf/2302.08091.pdf
        for subsample in [1]: # 
            print(subsample)
            subset_size = int(len(train_X) * subsample)
            subset_indices = np.random.choice(len(train_X), subset_size, replace=False)

            subsample_train_X = train_X[subset_indices]
            subsample_train_y = train_y[subset_indices]
            
            try:
                clf.fit(subsample_train_X, subsample_train_y)

                # Test score
                predict = clf.predict(test_X)
                class_report_dict = classification_report(test_y, predict, output_dict=True,labels=np.unique(test_y));
                report = pd.DataFrame.from_dict(class_report_dict).T.reset_index()
                report = report.rename(columns={"index":"class"})
            except:
                report = pd.DataFrame()
            report["kfold"] = i
            report["subsample_percent"] = subsample
            full_report_df = pd.concat([full_report_df, report])

        # Save last predictions
        full_predict.extend(predict)
        full_y.extend(test_y)

    # Add final "average" test score across all the folds
    class_report_dict = classification_report(full_predict, full_y, output_dict=True,labels=np.unique(full_predict))
    report = pd.DataFrame.from_dict(class_report_dict).T.reset_index()
    report = report.rename(columns={"index":"class"})
    report["kfold"] = "Full average"
    report["subsample_percent"] = 1
    full_report_df = pd.concat([full_report_df, report]) 
    full_report_df.to_csv(f"./data/{med_class_name}/baseline/{date}_kfold_{model_name}_{target_col}_{features}.csv")
    

def objective(trial: optuna.Trial, train_df, valid_df, n_labels):  
    
    """
    For Huggingface Trainer
    model = BertForSequenceClassification.from_pretrained("/wynton/group/ichs/shared_models/ucsf-bert", local_files_only=True)      
    
    model_args = {
        "learning_rate": trial.suggest_loguniform('learning_rate', low=1e-5, high=1e-4),
        "weight_decay":trial.suggest_loguniform('weight_decay', low=1e-5, high=1e-4),   
        "num_train_epochs":trial.suggest_int('num_train_epochs', low = 3, high = 4),
        "per_device_train_batch_size":8, 
        "per_device_eval_batch_size":8, 
    }
    """
    model_args = ClassificationArgs(learning_rate= trial.suggest_float('learning_rate', low=1e-5, high=1e-4), 
                                    weight_decay=trial.suggest_float('weight_decay', low=1e-5, high=1e-4),   
                                    num_train_epochs=5, 
                                    train_batch_size=16,
                                    eval_batch_size=16,
                                    use_early_stopping=True,
                                    output_dir="./data/ncdmard/ucsfbert",
                                    overwrite_output_dir=True,
                                    save_eval_checkpoints=False,
                                    save_model_every_epoch=False,
                                    silent=True,
                                    sliding_window=True
                                   )
    print(n_labels)
    model = ClassificationModel(model_type="bert", 
                                    model_name="/wynton/protected/project/shared_models/ucsf-bert", 
                                    tokenizer_type = BertTokenizer,
                                    tokenizer_name = "/wynton/protected/project/shared_models/ucsf-bert",
                                    num_labels=n_labels, 
                                    use_cuda=True,
                                #cuda_device=1,
                                   args=model_args)
        
    model.train_model(train_df);
    
    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(valid_df)
    
    return result["eval_loss"]

## Experiment params
 #"prev_medication_clean" #"medication_stopped_clean" #"medicationname_generic_clean"
#device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    
## Load annotations
def bert_run(target_col,
            date = "2023-10-11",
            model_name = "ucsfbert",
             med_class_name = "contraceptives"):
    annot_med = pd.read_parquet(f"./data/{med_class_name}/annotated_medications.parquet.gzip")
    annot_med = annot_med[annot_med["curr_med_change"]]
    annot_med = medswitch.cleanNotes(annot_med)
    print(annot_med["note_tokens"].describe())
    

    # Shuffle values
    benchmark_df = annot_med[["patientdurablekey",
                              "prev_medication", 
                              "mapped_med_generic_clean",
                              "next_medication",
                             "note_text_clean"]].sample(frac=1)
    print(benchmark_df.head())

    # Get KFold splits (don't need bow but borrowing from bow kfold split)
    bow, tfidf, y = preprocess.split_input_label(annot_med, text_col="note_text", target=target_col)
    groups = list(benchmark_df["patientdurablekey"])
    kfolds_bert = preprocess.kfold_split(X=bow.values,
                                 y=y[0], 
                                 pids=groups, 
                                 n_splits=5, 
                                 **{"train_size":0.875})

    # Testing on best model
    # With subsampling for training data
    full_report_df = pd.DataFrame()
    ucsf_bert_path="/wynton/group/ichs/shared_models/ucsf-bert"
    full_predict = []
    full_y = []

    for i, kfold in enumerate(kfolds_bert):
        # if i in [0,1]: continue

        train_pts, valid_pts, test_pts = kfold["p_splits"]
        print("Kfold "+str(i))

        # Get training values
        benchmark_df = annot_med[["note_text", target_col, "patientdurablekey"]]
        benchmark_df["labels"] = pd.factorize(benchmark_df[target_col])[0]
        benchmark_df["text"] = benchmark_df["note_text"]

        train_df = benchmark_df[benchmark_df["patientdurablekey"].isin(train_pts)][["text", "labels"]]
        valid_df = benchmark_df[benchmark_df["patientdurablekey"].isin(valid_pts)][["text", "labels"]]
        test_df = benchmark_df[benchmark_df["patientdurablekey"].isin(test_pts)][["text", "labels"]]

        n_labels = benchmark_df["labels"].nunique()

        # Grid search for hyperparameters
        obj_func = lambda trial: objective(trial, train_df, valid_df, n_labels)
        study = optuna.create_study(study_name='ucsfbert-search', direction='minimize') 
        study.optimize(func=obj_func, n_trials=5)
        print(study.best_params)

        # init new model arguments
        model_args = ClassificationArgs(learning_rate= float(study.best_params['learning_rate']),
                                        weight_decay=float(study.best_params['weight_decay']),   
                                        num_train_epochs=5,
                                        train_batch_size=32,
                                        eval_batch_size=32,
                                        use_early_stopping=True,
                                        output_dir=f"./data/{med_class_name}/baseline/ucsfbert/kfold_{i}_model",
                                        tensorboard_dir=f"./data/{med_class_name}/baseline/ucsfbert/",
                                        overwrite_output_dir=True,
                                        save_eval_checkpoints=False,
                                        save_model_every_epoch=False,
                                        silent=True,
                                       )

        # Subsample training dataset to simulate few-shot experiments (100%, 50%, 25%, 10%, 5%, 1%) 
        # based on https://arxiv.org/pdf/2302.08091.pdf
        for subsample in [1]:
            print(subsample)

            # subsample
            subsample_train_df = train_df.sample(frac=subsample)

           # Reset model
            model = ClassificationModel(model_type="bert", 
                                        model_name=ucsf_bert_path, 
                                        tokenizer_type = BertTokenizer,
                                        tokenizer_name = ucsf_bert_path,
                                        num_labels=n_labels, 
                                        use_cuda=True,
                                        #cuda_device=1,
                                       args=model_args)

            model.train_model(subsample_train_df.reset_index(drop=True));

            # Test score
            preds, prob = model.predict(list(test_df.reset_index(drop=True)["text"]))

            class_report_dict = classification_report(test_df["labels"], preds, 
                                                        output_dict=True,labels=np.unique(test_df["labels"]));
            report = pd.DataFrame.from_dict(class_report_dict).T.reset_index()
            report = report.rename(columns={"index":"class"})
            report["kfold"] = i
            report["subsample_percent"] = subsample
            full_report_df = pd.concat([full_report_df, report])

        full_report_df.to_csv(f"./data/{med_class_name}/baseline/{date}_kfold_{model_name}_{target_col}.csv")

        # Save last predictions
        full_predict.extend(preds)
        full_y.extend(list(test_df["labels"]))

    # Add final "average" test score across all the folds
    class_report_dict = classification_report(full_predict, full_y, output_dict=True,labels=np.unique(full_predict))
    report = pd.DataFrame.from_dict(class_report_dict).T.reset_index()
    report = report.rename(columns={"index":"class"})
    report["kfold"] = "Full average"
    report["subsample_percent"] = 1
    full_report_df = pd.concat([full_report_df, report])

    full_report_df.to_csv(f"./data/{med_class_name}/baseline/{date}_kfold_{model_name}_{target_col}.csv")


def full_bert_baseline():
    date = "2023-11-20"
    
    for target in ["mapped_med_generic_clean","prev_medication", "next_medication"]: #
        for med_class_name in ["contraceptives"]: #TODO: , "ncdmard" 
            bert_run(target_col = target,
                        date = date,
                        model_name = "ucsfbert",
                        med_class_name = med_class_name)
            
def full_ml_baseline():
    date = "2024-03-31"

    for target in ["next_med_change", "next_medication","prev_medication", "mapped_med_generic_clean",]:
        for model_name in ["rf", "logreg"]:
            for feature_name in ["bow","tfidf"]: 
                for med_class_name in ["ncdmard"]: #"ncdmard", 
                    curr_meds_only = True if target in ["prev_medication", "mapped_med_generic_clean", "next_medication"] else False
                    
                    ml_baseline(target_col = target,
                                date = date,
                                model_name = model_name,
                                features = feature_name,
                                med_class_name = med_class_name,
                               curr_meds_only=curr_meds_only)     

if __name__ == "__main__":
    #full_bert_baseline()
    full_ml_baseline()   