# Pneumonia-Detection
   
<p align="left"><img src="https://user-images.githubusercontent.com/63839581/118911251-dbc43500-b960-11eb-9632-aeca696b6b43.png" width="400" height="350"></p>   
   
## Description
   
This repository aims to solve the problem of **Pneumonia Detection** via AI which is the **Kaggle RSNA Challenge**.   
We performed **preprocessing, training, visualization (GradCam) and detection** using the given dataset in Kaggle RSNA.   
<br></br>
## Requirement
    
Requires Python 3.6 and keras 2.2.4, tensorflow-gpu 1.14.0, implemented by Cumtom [Retinanet](https://github.com/fizyr/keras-retinanet).
   
```bash
$ pip install -r requirements.txt
```
   
## Custom Dataset   
   
1. First, edit the [SETTINGS.json](https://github.com/sckim0430/Pneumonia-Detection/blob/master/SETTINGS.json) file to suit your personal environment.
2. Save [Kaggle RSNA DataSet](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/) in ./data/ folder.   
3. Run [prepare_data.sh](https://github.com/sckim0430/Pneumonia-Detection/blob/master/prepare_data.sh).   

```bash
$ sh prepare_data.sh
```

## Train
   
Modify [Retinanet.py](https://github.com/sckim0430/Pneumonia-Detection/blob/master/src/train/Retinanet/Retinanet.py) or [train.sh](https://github.com/sckim0430/Pneumonia-Detection/blob/master/train.sh).


```bash
$ sh train.sh
```   
   
or   
   
```bash
$ python ./src/train/Retinanet/Retinanet.py
```

## Evaluation

Modify [Eval_Retinanet.py](https://github.com/sckim0430/Pneumonia-Detection/blob/master/evaluate.sh) or [evaluate.sh](https://github.com/sckim0430/Pneumonia-Detection/blob/master/evaluate.sh).
    
```bash
$ sh evaluate.sh
```   
      
or
   
```bash
$ python ./src/eval/Retinanet/Eval_Retinanet.py
```

## Visualization
   
Modify [GradCam_Retinanet.py](https://github.com/sckim0430/Pneumonia-Detection/blob/master/src/infer/GradCam_Retinanet.py) or [visualization.sh](https://github.com/sckim0430/Pneumonia-Detection/blob/master/evaluate.sh).
   
    
```bash
$ sh visualization.sh
```   
   
or   
   
```bash
$ python ./src/infer/GradCam_Retinanet.py
```      
    
    
<p align="left"><img src="https://user-images.githubusercontent.com/63839581/118926853-ea6c1580-b97b-11eb-8f91-4874638992d1.png" width="400" height="350"></p>   
    
     
     
## Contack
   
another0430@naver.com   
