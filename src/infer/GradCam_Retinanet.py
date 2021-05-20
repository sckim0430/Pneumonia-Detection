from keras import backend as K
import os
import sys
import json

with open("SETTINGS.json") as f: 
    SETTINGS_JSON = json.load(f) 

sys.path.append("../../models/RetinaNet/")

from keras_retinanet.models import load_model
from keras_retinanet import utils
from keras.preprocessing import image
import numpy as np  
import cv2


MODEL_PATH = []

for model_file in os.listdir(SETTINGS_JSON['RETINANET_MODEL_DIR']):
    if os.path.splitext(model_file)[1] == '.h5':
        MODEL_PATH.append(os.path.join(SETTINGS_JSON['RETINANET_MODEL_DIR'],model_file))

IMAGE_PATH = SETTINGS_JSON['TEST_IMAGES_DIR']
RESULT_PATH = SETTINGS_JSON['HEATMAP_DIR']

backbone_name=MODEL_PATH[0].split('/')[-1].split('_')[0]
model = load_model(str(MODEL_PATH[0]),backbone_name=str(backbone_name),convert=True)

output = np.max(model.output[1][0]) #[1] : mean scores, [0] : mean class

last_conv_layer = model.get_layer('res5c_branch2c')

grads = K.gradients(output,last_conv_layer.output)[0]

pooled_grads = K.mean(grads,axis=(0,1,2))

iterate = K.function([model.input],[pooled_grads,last_conv_layer.output[0]])

for img_path in os.listdir(IMAGE_PATH):
    img = image.load_img(IMAGE_PATH+img_path,target_size=(384,384))
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    x= utils.image.preprocess_image(x)
    
    pooled_grads_value,conv_layer_output_value = iterate([x])

    for i in range(2048):
        conv_layer_output_value[:,:,i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value,axis=-1)
    
    heatmap /= np.max(heatmap)    

    heatmap = np.maximum(heatmap,0)
    
    img_cv = cv2.imread(IMAGE_PATH+img_path,cv2.IMREAD_COLOR)

    heatmap = cv2.resize(heatmap,(img_cv.shape[1],img_cv.shape[0]))
    
    heatmap = np.uint8(255*heatmap)
    #######################################################################
    total_num = len(heatmap)*len(heatmap[0])
    bins = np.arange(0,255,1)
    hist, bins = np.histogram(heatmap,bins)
    
    accumul_weight = []
    previous = 0
    for i in hist:
        previous += i
        accumul_weight.append(previous)
    
    for index,acc in enumerate(accumul_weight):
        if acc > total_num*0.7:
            thres = index
            break
    
    roi_index = []
    
    flatten_heatmap = heatmap.flatten()

    for index,val in enumerate(flatten_heatmap):
        if val <= thres:
            roi_index.append(index)

    roi_index_x = []
    roi_index_y = []

    for index,val in enumerate(roi_index):
        roi_index_y.append(val/len(heatmap))
        roi_index_x.append(val%len(heatmap))
        
    heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
       
    heatmap[roi_index_y,roi_index_x,:] = 0
    #######################################################################
    new_img = heatmap*0.4 + img_cv
    
    cv2.imwrite(RESULT_PATH+img_path,new_img)