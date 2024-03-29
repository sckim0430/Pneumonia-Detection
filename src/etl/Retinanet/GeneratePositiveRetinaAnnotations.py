###########
# IMPORTS #
###########
import pandas as pd
import numpy as np
import os
import json

with open("SETTINGS.json") as f:
    SETTINGS_JSON = json.load(f)

#############
# FUNCTIONS #
#############
##################################################################################################################################


def generate_annotations(folds_df,
                         data_dir,
                         save_annotations_dir,
                         save_validation_dir,
                         percent_normal,
                         fold=None):
    # Generate annotations for keras-retinanet for each fold
    # if fold is not None:
    #     folds = fold
    #     if type(folds) != list: folds = [folds]
    # else:
    #     folds = np.unique(folds_df.fold)
    for each_fold in range(SETTINGS_JSON['RETINANET_FOLDS']):
        tmp_train_df = folds_df[folds_df.fold != each_fold]
        tmp_valid_df = folds_df[folds_df.fold == each_fold]
        tmp_train_df["filepath"] = [os.path.join(
            data_dir, "{}.png".format(pid)) for pid in tmp_train_df.patientId]
        tmp_valid_df["filepath"] = [os.path.join(
            data_dir, "{}.png".format(pid)) for pid in tmp_valid_df.patientId]

        tmp_train_df["Target"] = ["Pneumonia" if target ==
                                  1 else None for target in tmp_train_df.Target]

        tmp_train_df_pos = tmp_train_df[tmp_train_df.Target == "Pneumonia"]
        pos_frac = 1. - percent_normal
        num_unique_pos = len(np.unique(tmp_train_df_pos.patientId))

        tmp_train_df_neg = tmp_train_df[tmp_train_df["Target"] != "Pneumonia"]
        tmp_train_df_neg = tmp_train_df_neg.sample(
            n=int(num_unique_pos*percent_normal/pos_frac))

        tmp_train_df = tmp_train_df_pos.append(tmp_train_df_neg)
        #tmp_train_df = tmp_train_df.sample(frac=1)
        tmp_train_df = tmp_train_df[[
            "filepath", "x1", "y1", "x2", "y2", "Target"]]
        tmp_valid_df = tmp_valid_df[[
            "filepath", "x1", "y1", "x2", "y2", "Target"]]
        # Leave validation as positives only
        # Evaluation script will use the full validation set
        tmp_valid_df["Target"] = ["Pneumonia" if target ==
                                  1 else None for target in tmp_valid_df.Target]
        tmp_train_df.to_csv(save_annotations_dir.format(
            each_fold), header=False, index=False)
        tmp_valid_df.to_csv(save_validation_dir.format(
            each_fold), header=False, index=False)

##################################################################################################################################

##########
# SCRIPT #
##########


# Add bbox labels to folds
labels_df = pd.read_csv(SETTINGS_JSON['TRAIN_CSV_DIR'])
folds_df = pd.read_csv(SETTINGS_JSON['RETINANET_STRATIFIED_FOLDS_CSV_PATH'])
folds_df = folds_df.merge(labels_df, on="patientId")

folds_df["x1"] = folds_df["x"]
folds_df["y1"] = folds_df["y"]
folds_df["x2"] = folds_df["x"] + folds_df["w"]
folds_df["y2"] = folds_df["y"] + folds_df["h"]

folds_df = folds_df.fillna(8888888)

folds_df["x1"] = folds_df.x1.astype("int32").astype("str")
folds_df["y1"] = folds_df.y1.astype("int32").astype("str")
folds_df["x2"] = folds_df.x2.astype("int32").astype("str")
folds_df["y2"] = folds_df.y2.astype("int32").astype("str")

folds_df = folds_df.replace({"8888888": None})

del folds_df["x"], folds_df["y"]

data_dir = SETTINGS_JSON['TRAIN_IMAGES_DIR']
save_annotations_dir = SETTINGS_JSON['RETINANET_TRAIN_CSV_DIR']
save_validation_dir = SETTINGS_JSON['RETINANET_VALIDATION_CSV_DIR']

generate_annotations(folds_df, data_dir, save_annotations_dir,
                     save_validation_dir, 0, None)
