# Specify GPU
import os
import scipy.misc
import numpy as np
from scipy.ndimage.interpolation import zoom
from sklearn.metrics import roc_auc_score
import pandas as pd
import re
import glob
from sklearn.metrics import precision_score
from keras_retinanet.utils import evaluate_with_kaggle_metric
from keras_retinanet.models import load_model
import os
import json

with open("SETTINGS.json") as f:
    SETTINGS_JSON = json.load(f)

import sys
sys.path.append("../../models/RetinaNet/")


# Specify GPU (if needed)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def KaggleTestSetEvaluate(model_save_path,
                          backbone_name,
                          test_annotations,
                          image_input_size,
                          data_dir,
                          max_detections=5,
                          score_threshold=np.linspace(0.05, 0.95, 19)):

    model = load_model(model_save_path,
                       backbone_name=backbone_name,
                       convert=True)

    test_annotations_df = pd.read_csv(test_annotations)
    test_annotations_df.columns = ["patientID", "Target", "x", "y", "w", "h"]
    test_annotations_df["filepath"] = [os.path.join(
        data_dir, patientID)+'.png' for patientID in test_annotations_df.patientID]

    # Lists for Kaggle mAP over all images
    list_of_metrics = []
    list_of_threshs = []
    list_of_imagids = []
    # Lists for Kaggle mAP over positive images
    list_of_metrics_positives_only = []
    list_of_threshs_positives_only = []
    list_of_imagids_positives_only = []

    list_of_scores = []
    list_of_labels = []
    list_of_pids = []

    num_images = len(np.unique(test_annotations_df.filepath))

    for index, each_img in enumerate(np.unique(test_annotations_df.filepath)):
        sys.stdout.write("Predicting: {}/{} ...\r".format(index+1, num_images))
        sys.stdout.flush()
        test_id = os.path.splitext(each_img.split("/")[-1])[0]
        # Read as BGR
        tmp_img = scipy.misc.imread(each_img, mode="RGB")
        tmp_img = tmp_img[..., ::-1]
        # Image will always be square
        scale = float(image_input_size) / tmp_img.shape[0]
        if scale != 1.:
            tmp_img = zoom(tmp_img, [scale, scale, 1.],
                           order=1, prefilter=False)
        tmp_img = evaluate_with_kaggle_metric.preprocess_input(
            tmp_img, backbone_name)

        prediction = model.predict_on_batch(np.expand_dims(tmp_img, axis=0))

        # Get ground truth for image
        tmp_df = test_annotations_df[test_annotations_df.filepath == each_img]
        if int(tmp_df.Target.iloc[0]) == 0:
            gt_bboxes = np.empty((0, 4))
        else:
            gt_bboxes = [np.asarray((row.x, row.y, row.w, row.h), dtype=np.float)
                         for rownum, row in tmp_df.iterrows()]
            gt_bboxes = np.asarray(gt_bboxes)
        # Save values for classifier evaluation
        list_of_scores.append(np.max(prediction[1][0]))
        list_of_labels.append(tmp_df.Target.iloc[0])
        #list_of_views.append(tmp_df.view.iloc[0])
        list_of_pids.append(test_id)
        for each_thres in score_threshold:
            scores = prediction[1][0]
            bboxes = prediction[0][0]
            # Ensure that scores are sorted in descending order
            sorted_indices = np.argsort(scores)[::-1]
            scores = np.asarray(scores[sorted_indices])
            bboxes = np.asarray(bboxes[sorted_indices])
            # Get boxes greater than threshold
            detected = scores >= each_thres
            # Limit number of boxes to max_detections
            if np.sum(detected) > max_detections:
                detected[max_detections:] = False
            scores = np.asarray(scores[detected])
            bboxes = np.asarray(bboxes[detected])
            list_of_bboxes = []
            # Rescale boxes
            bboxes = [box / scale for box in bboxes]
            for each_box in bboxes:
                x1 = each_box[0]
                y1 = each_box[1]
                ww = each_box[2] - each_box[0]
                hh = each_box[3] - each_box[1]
                list_of_bboxes.append((np.asarray((x1, y1, ww, hh))))
            box_array = np.asarray(list_of_bboxes, dtype=np.float)
            # Calculate metric
            # print(gt_bboxes)
            # print(box_array)
            mapiou = evaluate_with_kaggle_metric.mAP_IoU(
                gt_bboxes, box_array, scores)
            # Positives only
            if len(gt_bboxes) != 0:
                list_of_metrics_positives_only.append(mapiou)
                list_of_threshs_positives_only.append(each_thres)
                list_of_imagids_positives_only.append(test_id)
            # All images
            list_of_metrics.append(mapiou)
            list_of_threshs.append(each_thres)
            list_of_imagids.append(test_id)

    results_df = pd.DataFrame({"patientId": list_of_imagids,
                               "mAP":       list_of_metrics,
                               "threshold": list_of_threshs})
    results_df = results_df[["mAP", "threshold"]].groupby(
        "threshold").mean().reset_index()

    results_df_positives_only = pd.DataFrame({"patientId": list_of_imagids_positives_only,
                                              "mAP":       list_of_metrics_positives_only,
                                              "threshold": list_of_threshs_positives_only})
    results_df_positives_only = results_df_positives_only[[
        "mAP", "threshold"]].groupby("threshold").mean().reset_index()

    #
    scores_df = pd.DataFrame({"patientId": list_of_pids,
                              "y_score":   list_of_scores,
                              "y_true":    list_of_labels})

    #auroc = roc_auc_score(scores_df.y_true, scores_df.y_score)
    scores = []

    for val in list_of_scores:
        if val > 0.5:
            scores.append(1)
        else:
            scores.append(0)

    auroc = precision_score(scores_df.y_true, scores)

    return np.max(results_df.mAP), np.max(results_df_positives_only.mAP), auroc


TEST_IMAGES_DIR = SETTINGS_JSON['TEST_IMAGES_DIR']

MODEL_PATH = []

for model_file in os.listdir(SETTINGS_JSON['RETINANET_MODEL_DIR']):
    if os.path.splitext(model_file)[1] == '.h5':
        MODEL_PATH.append(os.path.join(
            SETTINGS_JSON['RETINANET_MODEL_DIR'], model_file))

TEST_ANNOTATIONS = SETTINGS_JSON['TEST_CSV_DIR']

backbone_name = MODEL_PATH[0].split('/')[-1].split('_')[0]
image_min_side = 384

eval = KaggleTestSetEvaluate(str(MODEL_PATH[0]), str(
    backbone_name), TEST_ANNOTATIONS, image_min_side, TEST_IMAGES_DIR)

eval_result_of_all = eval[0]
eval_result_of_pos = eval[1]
eval_result_of_clf = eval[2]

print('result of all : {}'.format(eval_result_of_all))
print('result of pos : {}'.format(eval_result_of_pos))
print('result of clf : {}'.format(eval_result_of_clf))
