def objective(trial: optuna.Trial, train_df, valid_df):  
    
    """
    For Huggingface Trainer
    model = BertForSequenceClassification.from_pretrained("/wynton/group/ichs/shared_models/ucsf-bert", local_files_only=True)      
    
    model_args = {
        "learning_rate": trial.suggest_loguniform('learning_rate', low=4e-5, high=0.01),
        "weight_decay":trial.suggest_loguniform('weight_decay', 4e-5, 0.01),   
        "num_train_epochs":trial.suggest_int('num_train_epochs', low = 3, high = 4),
        "per_device_train_batch_size":8, 
        "per_device_eval_batch_size":8, 
        "use_early_stopping":True,
        "overwrite_output_dir":True,
        "output_dir":"./data/contraceptives/ucsfbert",
        "save_eval_checkpoints":False,
        "save_model_every_epoch":False
    }
    """
    model_args = ClassificationArgs(learning_rate= trial.suggest_float('learning_rate', low=4e-5, high=0.01), 
                                    weight_decay=trial.suggest_float('weight_decay', low=4e-5, high=0.01),   
                                    num_train_epochs=1, 
                                    train_batch_size=16,
                                    eval_batch_size=16,
                                    use_early_stopping=True,
                                    output_dir="./data/contraceptives/ucsfbert",
                                    overwrite_output_dir=True,
                                    save_eval_checkpoints=False,
                                    save_model_every_epoch=False,
                                    silent=True,
                                   )
    
    model = ClassificationModel(model_type="bert", 
                                    model_name="/wynton/group/ichs/shared_models/ucsf-bert", 
                                    tokenizer_type = BertTokenizer,
                                    tokenizer_name = "/wynton/group/ichs/shared_models/ucsf-bert",
                                    num_labels=n_labels, 
                                    use_cuda=True,
                                cuda_device=1,
                                   args=model_args)
        
    model.train_model(train_df);
    
    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(valid_df)
    
    return result["eval_loss"]
