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
import argparse
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from matplotlib.pyplot import imshow
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
K.tensorflow_backend.set_session(tf.Session(config=config))

#img-path
parser = argparse.ArgumentParser()

parser.add_argument('--dir',default='/home/hclee/test/Pneumonia.png')

args = parser.parse_args()

MODEL_PATH = []

for model_file in os.listdir(SETTINGS_JSON['RETINANET_MODEL_DIR']):
    if os.path.splitext(model_file)[1] == '.h5':
        MODEL_PATH.append(os.path.join(SETTINGS_JSON['RETINANET_MODEL_DIR'],model_file))

backbone_name=MODEL_PATH[0].split('/')[-1].split('_')[0]
model = load_model(str(MODEL_PATH[0]),backbone_name=str(backbone_name),convert=True)
output = np.max(model.output[1][0]) #[1] : mean scores, [0] : mean class

last_conv_layer = model.get_layer('res5c_branch2c')

grads = K.gradients(output,last_conv_layer.output)[0]

pooled_grads = K.mean(grads,axis=(0,1,2))

iterate = K.function([model.input],[pooled_grads,last_conv_layer.output[0]])

output_layer = model.get_layer('filtered_detections')
iterate2 = K.function([model.input],[output_layer.output[1][0]])

img = image.load_img(args.dir,target_size=(384,384))
x = image.img_to_array(img)
x = np.expand_dims(x,axis=0)
x= utils.image.preprocess_image(x)
    
pooled_grads_value,conv_layer_output_value = iterate([x])

for i in range(2048):
    conv_layer_output_value[:,:,i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value,axis=-1)
    
heatmap /= np.max(heatmap)    

heatmap = np.maximum(heatmap,0)
    
img_cv = cv2.imread(args.dir,cv2.IMREAD_COLOR)

heatmap = cv2.resize(heatmap,(img_cv.shape[1],img_cv.shape[0]))
    
heatmap = np.uint8(255*heatmap)

heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)

new_img = heatmap*0.4 + img_cv

pred_val = iterate2([x])
pred_val = pred_val[0][0]

cv2.imwrite('/home/hclee/test/Pneumonia_heatmap.png',new_img)

#################################################################################################
test_image = Image.open('/home/hclee/test/Pneumonia.png').convert('RGB')
draw = ImageDraw.Draw(test_image)
font = ImageFont.truetype("/home/hclee/.local/share/fonts/gulim.ttf",80)
draw.text((0,0),str(round(pred_val,2)),(255,255,255),font=font)
test_image.show()
#################################################################################################

test_image = Image.open('/home/hclee/test/Pneumonia_heatmap.png').convert('RGB')
draw = ImageDraw.Draw(test_image)
font = ImageFont.truetype("/home/hclee/.local/share/fonts/gulim.ttf",80)
draw.text((0,0),str(round(pred_val,2)),(255,255,255),font=font)
test_image.show()
