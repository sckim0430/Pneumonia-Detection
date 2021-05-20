###########
# IMPORTS #
###########

import pandas as pd 
import numpy as np 
import os, json 

with open("SETTINGS.json") as f: 
    SETTINGS_JSON = json.load(f)

#############
# FUNCTIONS #
#############

def assign_folds(orig_df,num_folds):
    df = orig_df.copy()
    k = np.arange(len(df['patientId']))
    k = np.random.permutation(k)

    folds = []
    for fold in range(num_folds):
        if fold == num_folds-1:
            folds.append(k[(len(df)/num_folds)*fold:])
            break

        folds.append(k[(len(df)/num_folds)*fold:(len(df)/num_folds)*(fold+1)])
    
    fold_counter =0

    df["fold"] = None  

    for fold in folds:
       df['fold'].iloc[fold] = fold_counter
       fold_counter += 1

    return df
##########
# SCRIPT #
##########

path_to_labels_file = SETTINGS_JSON['TRAIN_CSV_DIR']
labels_df = pd.read_csv(path_to_labels_file).drop_duplicates('patientId')
labels_df = labels_df.drop(['Target','x','y','w','h'],axis='columns')

folds_df = assign_folds(labels_df, 10)

folds_df.to_csv(SETTINGS_JSON['RETINANET_STRATIFIED_FOLDS_CSV_PATH'],index=False)

print(folds_df)


