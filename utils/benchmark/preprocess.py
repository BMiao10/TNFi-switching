import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def _extract_features(docs, vectorizer):
    """BOW or TFIDF"""
    X = vectorizer.fit_transform(docs)
    vector_df = pd.DataFrame(X.toarray(),columns=vectorizer.get_feature_names_out())

    return vector_df

def split_input_label(values_df, text_col, target):
    """
    Given single dataframe of note values and targets, split into inputs and labels
    """

    # input values (remove English stop words)
    bow_vectorizer = CountVectorizer(stop_words='english')
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8, min_df=5)

    bow = _extract_features(values_df[text_col], vectorizer=bow_vectorizer)
    tfidf = _extract_features(values_df[text_col], vectorizer=tfidf_vectorizer)

    # Load target labels
    y = pd.factorize(values_df[target])
    
    return bow, tfidf, y

def kfold_split_df(df,
                   group_col, 
                   n_splits=5,
                  random_state=0):
    """
    Splits dataframe into train/valid/test splits, stratified by group.

    Args:
        df (pandas.DataFrame): Dataframe containing features, labels, and group columns
        group (str): Column name for the group variable
        n_splits (int): Number of folds to generate
        random_state (int): sets random state for KFold and train_test_split

    Returns:
        List of dictionaries, each containing train/valid/test splits for a fold
    """

    # Initialize the GroupKFold object with the desired number of splits
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Get the values of the target and group columns
    group = np.array(df[group_col].unique())
    kfold_split = kf.split(group)

    folds = []
    for i, (group_train_index, group_test_index) in enumerate(kfold_split):
        train_pts = list(group[group_train_index])
        test_pts = list(group[group_test_index])
        
        # Get the indices for the training, validation, and testing sets
        train_pts, valid_pts = train_test_split(train_pts, test_size=len(test_pts), random_state=random_state)
        
        # Store the data for the current fold as a dictionary
        fold_dict = {}
        for fold_key, fold_pts in [("train",train_pts), 
                                     ("validation", valid_pts), 
                                     ("test", test_pts)]:
            fold_dict[fold_key] = df[df[group_col].isin(fold_pts)]
        
        folds.append(fold_dict)
        print('Made splits: %d / %d / %d' % (len(train_pts), len(valid_pts), len(test_pts)))
        
    return folds

def kfold_split(X,y, pids, n_splits, random_state=0, **kwargs):
    """
    Splits diagnosis data into train/valid/test splits without label leakage
    via patient_ids between 5 folds.
    
    Author: Irene Chen

    Steps:
     1. Split into 5 folds
     2. Create tr/tr/tr/vl/ts
     3. Save each fold
    """
    folds = list()

    # get all unique pids
    # split those into folds (70/10/20)
    # using pid folds, generate X, y for train/valid/test

    pids_unique = np.unique(pids)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for p_tr_idx, p_ts_idx in kf.split(pids_unique):
        p_tr_idx, p_vl_idx = train_test_split(p_tr_idx, random_state=random_state, **kwargs)

        p_tr = pids_unique[p_tr_idx]
        p_vl = pids_unique[p_vl_idx]
        p_ts = pids_unique[p_ts_idx]
        
        train_idx = np.array([i for i,j in enumerate(pids) if j in p_tr])
        valid_idx = np.array([i for i,j in enumerate(pids) if j in p_vl])
        test_idx = np.array([i for i,j in enumerate(pids) if j in p_ts])
        
        p_tr_all = np.array([j for i,j in enumerate(pids) if j in p_tr])
        p_vl_all = np.array([j for i,j in enumerate(pids) if j in p_vl])
        p_ts_all = np.array([j for i,j in enumerate(pids) if j in p_ts])
        
        train_X, valid_X, test_X = X[train_idx, :], X[valid_idx, :], X[test_idx, :]
        train_y, valid_y, test_y = y[train_idx], y[valid_idx], y[test_idx]

        print('Made splits: %d / %d / %d' % (len(train_idx), len(valid_idx), len(test_idx)))

        fold_dict = {
                    'p_splits': (p_tr_all, p_vl_all, p_ts_all),
                    'idx_splits': (train_idx, valid_idx,test_idx),
                    'X_vals': (train_X, valid_X, test_X),
                    'y_vals': (train_y, valid_y, test_y)
        }
        folds.append(fold_dict)

    return folds