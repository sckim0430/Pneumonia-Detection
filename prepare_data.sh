echo "Creating stratified K-fold cross-validation ..."
python src/etl/Retinanet/AssignCVFolds.py

echo "Generating Annotations for training keras-retinaet ..."
python src/etl/Retinanet/GeneratePositiveRetinaAnnotations.py

echo "DONE !"