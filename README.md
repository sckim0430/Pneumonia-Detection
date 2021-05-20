# Pneumonia-Detection
   
<p align="left"><img src="https://user-images.githubusercontent.com/63839581/118911251-dbc43500-b960-11eb-9632-aeca696b6b43.png" width="400" height="350"></p>   
   
## Description
   
이 저장소는 **Kaggle RSNA Challenge**인 AI를 통한 **Pneumonia Detection** 문제를 해결하는 것을 목표로 합니다.   
Kaggle RSNA에서 주어진 데이터셋을 활용하여 **전처리, 학습, 시각화(GradCam) 및 검출**을 진행했습니다.   
   
## Requirement
    
Python 3.6 및 keras 2.2.4, tensorflow-gpu 1.14.0이 필요하며, [Retinanet](https://github.com/fizyr/keras-retinanet)을 Cumtom하여 구현했습니다.   
   
```bash
$ pip install -r requirements.txt
```
   
## Custom Dataset   
   
1. 먼저 [SETTINGS.json](https://github.com/sckim0430/Pneumonia-Detection/blob/master/SETTINGS.json)파일을 개인 환경에 맞게 수정해줍니다.   
2. ./data/ 폴더에 [Kaggle RSNA DataSet](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/)을 저장합니다.   
3. [prepare_data.sh](https://github.com/sckim0430/Pneumonia-Detection/blob/master/prepare_data.sh)을 실행합니다.   

```bash
$ sh prepare_data.sh
```

## Train
   
[Retinanet.py](https://github.com/sckim0430/Pneumonia-Detection/blob/master/src/train/Retinanet/Retinanet.py)를 수정하거나 [train.sh](https://github.com/sckim0430/Pneumonia-Detection/blob/master/train.sh)을 실행합니다.

```bash
$ sh train.sh
```   
   
or   
   
```bash
$ python ./src/train/Retinanet/Retinanet.py
```

## Evaluation

[Eval_Retinanet.py](https://github.com/sckim0430/Pneumonia-Detection/blob/master/evaluate.sh)을 수정하거나 [evaluate.sh](https://github.com/sckim0430/Pneumonia-Detection/blob/master/evaluate.sh)을 실행합니다.   
   
```bash
$ sh evaluate.sh
```   
      
or
   
```bash
$ python ./src/eval/Retinanet/Eval_Retinanet.py
```

## Visualization
   
[GradCam_Retinanet.py](https://github.com/sckim0430/Pneumonia-Detection/blob/master/src/infer/GradCam_Retinanet.py)을 수정하거나 [visualization.sh](https://github.com/sckim0430/Pneumonia-Detection/blob/master/visualization.sh)을 실행합니다.   
   
```bash
$ sh visualization.sh
```   
   
or   
   
```bash
$ python ./src/infer/GradCam_Retinanet.py
```      
   
<p align="left"><img src="https://user-images.githubusercontent.com/63839581/118926853-ea6c1580-b97b-11eb-8f91-4874638992d1.png" width="400" height="350"></p>   
   
