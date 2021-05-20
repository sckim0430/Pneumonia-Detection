echo "Evaluate Retinanet Model"
python src/eval/Retinanet/Eval_Retinanet.py

echo "Make Heatmap with Retinanet Model"
python src/eval/Retinanet/GradCam_Retinanet.py