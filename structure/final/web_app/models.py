from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, \
    GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier 
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


def run_model(model_name, X, y):
    
    if model_name == 'LR':
        model = LogisticRegression().fit(X, y)
    elif model_name == 'SGD':
        model = SGDClassifier().fit(X, y)
    elif model_name == 'DT':
        model = DecisionTreeClassifier().fit(X, y)
    elif model_name == 'RF':
        model = RandomForestClassifier().fit(X, y)
    elif model_name == 'AdaBoost':
        model = AdaBoostClassifier().fit(X, y)
    elif model_name == 'GBoost':
        model = GradientBoostingClassifier().fit(X, y)
    elif model_name == 'XGB':
        model = XGBClassifier().fit(X, y)
    elif model_name == 'MLP':
        model = MLPClassifier().fit(X, y)
    elif model_name == 'NB':  
        model = GaussianNB().fit(X, y)
    elif model_name == 'KNN':  
        model = KNeighborsClassifier().fit(X, y)
    elif model_name == 'LGBM':  
        model = LGBMClassifier().fit(X, y)
    elif model_name == 'CatBoost':  
        model = CatBoostClassifier(verbose=0).fit(X, y)  # `verbose=0` suppresses output
    
        
    return model


def modeling(model_names, X_train, y_train, X_val, y_val):
    
    results = {}
    
    for model_name in model_names:
        print(f"Training model: {model_name}")
        
        # Train model
        model = run_model(model_name, X_train, y_train)
        
        # Predict on train and validation data
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        # Calculate metrics for the training set
        train_precision = precision_score(y_train, y_train_pred, average='macro')
        train_recall = recall_score(y_train, y_train_pred, average='macro')
        train_f1 = f1_score(y_train, y_train_pred, average='macro')
        
        # Calculate metrics for the validation set
        val_precision = precision_score(y_val, y_val_pred, average='macro')
        val_recall = recall_score(y_val, y_val_pred, average='macro')
        val_f1 = f1_score(y_val, y_val_pred, average='macro')
        
        # Save results
        results[model_name] = {
            'train_precision': train_precision,
            'val_precision': val_precision,
            'train_recall': train_recall,
            'val_recall': val_recall,
            'train_macro_f1': train_f1,
            'val_macro_f1': val_f1
        }
        
        print(f"{model_name} - Train: Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, Macro F1: {train_f1:.4f}")
        print(f"{model_name} - Validation: Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, Macro F1: {val_f1:.4f}\n")
    
    return results
