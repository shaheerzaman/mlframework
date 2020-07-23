import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib

import dispatcher

TRAINING_DATA = '../input/train_folds.csv'
TEST_DATA = '../input/test.csv'

MODEL = 'extratrees'

FOLD_MAPPING={
    0:[1, 2, 3, 4], 
    1:[0, 2, 3, 4], 
    2:[0, 1, 3, 4], 
    3:[0, 1, 2, 4], 
    4:[0, 1, 2, 3]
}

def run(fold):
    df = pd.read_csv(TRAINING_DATA)
    df_test = pd.read_csv(TEST_DATA)
    train_df = df[df.kfold.isin(FOLD_MAPPING.get(fold))].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)

    y_train = train_df.target.values
    y_valid = valid_df.target.values

    train_df = train_df.drop(['kfold','target', 'id'], axis=1)
    valid_df = valid_df.drop(['kfold', 'target', 'id'],axis=1)
    
    valid_df = valid_df[train_df.columns]
    
    label_encoders = {}
    for c in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        train_df.loc[:, c] = train_df.loc[:, c].astype(str).fillna('None')
        valid_df.loc[:, c] = valid_df.loc[:, c].astype(str).fillna('None')
        df_test.loc[:, c] = df_test.loc[:, c].astype(str).fillna('None')

        lbl.fit(train_df[c].values.tolist() + 
                valid_df[c].values.tolist() + 
                df_test[c].values.tolist())
        train_df.loc[:, c] = lbl.transform(train_df.loc[:, c].values.tolist())
        valid_df.loc[:, c] = lbl.transform(valid_df.loc[:, c].values.tolist())
        label_encoders[c] = lbl

    
    #data is ready to train
    clf = dispatcher.MODELS[MODEL]
    clf.fit(train_df, y_train)
    preds = clf.predict_proba(valid_df)[:, 1]
    score = metrics.roc_auc_score(y_valid, preds)

    joblib.dump(label_encoders, f'../models/{MODEL}_{fold}_label_encoder.pkl')
    joblib.dump(clf, f'../models/{MODEL}_{fold}.pkl')
    joblib.dump(train_df.columns, f'../models/{MODEL}_{fold}_columns.pkl')

    return score

if __name__ == '__main__':
    #fold1_score = run(fold=0)
    fold1_score = run(fold=1)
    fold2_score = run(fold=2)
    fold3_score = run(fold=3)
    fold4_score = run(fold=4)

    print(f'Score for fold 1: {fold1_score}')
    print(f'Score for fold 2: {fold1_score}')
    print(f'Score for fold 3: {fold1_score}')
    print(f'Score for fold 4: {fold1_score}')

