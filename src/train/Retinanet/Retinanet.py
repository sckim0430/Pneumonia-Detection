import os, json

with open("SETTINGS.json") as f: 
    SETTINGS_JSON = json.load(f) 

def train(folds,backbone, 
          gpu=0, 
          stratified_folds=SETTINGS_JSON['RETINANET_STRATIFIED_FOLDS_CSV_PATH'], 
          batch_size=8,
          data_dir=SETTINGS_JSON['TRAIN_IMAGES_DIR'], 
          steps=3000,
          epochs=25,
          snapshot_path=SETTINGS_JSON['RETINANET_SNAPSHOT_DIR'],
          image_min_side=384,
          image_max_side=384,
          annotations_folder=SETTINGS_JSON['RETINANET_TRAIN_CSV_DIR'], 
          classes_file_path=SETTINGS_JSON['RETINANET_CLASS_CSV_DIR'], 
          val_annotations=SETTINGS_JSON['RETINANET_VALIDATION_CSV_DIR']): 
  TRAIN_KAGGLE_PATH = SETTINGS_JSON['RETINANET_TRAIN_FILE_DIR']
  snapshot_path = snapshot_path.format(folds)
  annotations_folder = annotations_folder.format(folds) 
  val_annotations = val_annotations.format(folds) 
  command  = "python {} --backbone {} --batch-size {}".format(TRAIN_KAGGLE_PATH, backbone, batch_size) 
  command += " --gpu {} --stratified_folds {} --fold {} --data_dir {}".format(gpu, stratified_folds, fold, data_dir) 
  command += " --steps {} --epochs {} --snapshot-path {}".format(steps, epochs, snapshot_path)
  command += " --image_min_side {} --image_max_side {} csv {}".format(image_min_side, image_max_side, annotations_folder)
  command += " {} --val-annotations {}".format(classes_file_path, val_annotations)
  os.system(command) 

folds = SETTINGS_JSON['RETINANET_FOLDS']

for fold in range(folds):
  if fold%2 == 0:
    train(fold,"resnet152")
  else:
    train(fold,"resnet101")

