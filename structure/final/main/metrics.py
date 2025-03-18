from sklearn.metrics import classification_report
import pandas as pd

def metrics(y_train, pred_train , y_val, pred_val):
    '''
    Input: actual values of X and y train and their predictions
    Outputs: Classification Report for Train and Validation Data
    '''
    print('______________________________________________________________________')
    print('                                TRAIN                                 ')
    print('----------------------------------------------------------------------')
    print(classification_report(y_train, pred_train))



    print('______________________________________________________________________')
    print('                                VALIDATION                                 ')
    print('----------------------------------------------------------------------')
    print(classification_report(y_val, pred_val))


def metrics2(models, model_names):
    '''
    Input: 
        models:list of Models to be evaluated
        model_names: the names of said models
    Outputs: DataFrame with metrics of the models and their training time
    '''

    # Metrics
    f1macro_train = []
    f1macro_val = []

    precision_train = []
    precision_val = []

    recall_train = []
    recall_val = []

    times = []

    # For each Model append Metrics
    for model in models: 

        f1macro_train.append(model['avg_f1_train'])
        f1macro_val.append(model['avg_f1_val'])

        precision_train.append(model['avg_precision_train'])
        precision_val.append(model['avg_precision_val'])

        recall_train.append(model['avg_recall_train'])
        recall_val.append(model['avg_recall_val'])

        times.append(model['avg_time'])  

    # Save results in a Dataframe
    results = pd.DataFrame(
        {
            "Train F1 macro": f1macro_train,
            "Validation F1 macro": f1macro_val,
            "Precision Train": precision_train,
            "Precision Validation": precision_val,
            "Recall Train": recall_train,
            "Recall Validation": recall_val,
            "Time": times
        },
        index = model_names)
    
    return results.T
