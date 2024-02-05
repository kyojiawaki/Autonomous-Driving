# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Python 2と3の両方で動作するためのfutureモジュールのimport
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 以下、必要なライブラリやモジュールのimport
import os.path
import re
import sys
import cv2
import time
import datetime
import glob
import numpy as np
import tflite_runtime.interpreter as tflite # TensorFlow Liteランタイム用のモジュール
from six.moves import urllib
from PIL import Image, ImageFilter, ImageChops
from operator import itemgetter
import RPi.GPIO as GPIO
import subprocess
import picar_4wd as fc
#################追加部分(モジュールインポート)###############
from torchvision import transforms
from awaki_detect.models.common import DetectMultiBackend
from awaki_detect.utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    non_max_suppression,
    scale_boxes,
)
from awaki_detect.utils.torch_utils import select_device, smart_inference_mode
############################################################

#tflite setup
input_mean = 127.5
input_std = 127.5
#location of model file ####modelfile = "converted_model.tflite"
classname = ['left', 'right', 'straight']
##################追加部分(モデル選択)#################
rccar_model_dir = "awaki_model/rccar"
detect_model_dir = "awaki_model/detect"

# モデルファイルのリストを表示
print("Choose a rccar model:")
model_files = os.listdir(rccar_model_dir)
for model_file in model_files:
    print(" - "+model_file)
# ユーザーにrccarモデルの選択を促す
selected_model = input("Enter the rccar model name >> ")
# 選択されたrccarモデルのパスを作成
rccar_modelfile = os.path.join(rccar_model_dir, selected_model)
# 選択されたrccarモデルが存在するか確認
if not os.path.exists(rccar_modelfile):
    print("Selected model not found!!!!!!!! : ", rccar_modelfile)
    sys.exit(1)

# モデルファイルのリストを表示
print("Choose a detect model:")
model_files = os.listdir(detect_model_dir)
for model_file in model_files:
    print(" - "+model_file)
# ユーザーにdetectモデルの選択を促す
selected_model = input("Enter the detect model name >> ")
# 選択されたdetectモデルのパスを作成
detect_modelfile = os.path.join(detect_model_dir, selected_model)
# 選択されたrccarモデルが存在するか確認
if not os.path.exists(detect_modelfile):
    print("Selected model not found!!!!!!!! : ", detect_modelfile)
    sys.exit(1)
######################################################

#output files
#name = sys.argv[1]
#imagedir = "/home/pi/pitank/auto/image/" + name
#logfile = "/home/pi/pitank/auto/image/" + name + ".log"
name = sys.argv[1]
imagedir = "image/"+name
logfile = "image/"+name+".log"

testimage = False

#change speed for straight (value[0]) or left/right (value[1] and value[2])
#value = [30, 10, 98]
value = [1, 1, 100]
interval = 0.1

if not os.path.exists(imagedir):
    os.makedirs(imagedir)
    dirname = imagedir + '/left'
    os.makedirs(dirname)
    dirname = imagedir + '/right'
    os.makedirs(dirname)
    dirname = imagedir + '/straight'
    os.makedirs(dirname)

def getPicture(currenttime, cap):
    jpegfile = imagedir + '/' + currenttime + '.jpeg'
    _, img = cap.read()
    cv2.imwrite(jpegfile, img)
    return jpegfile

def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

def classify_image(interpreter, image):
    """Returns a sorted array of classification results."""
    input_details = interpreter.get_input_details()
    #print(input_details)
    floating_model = input_details[0]['dtype'] == np.float32
    #print("floating", floating_model)
    input_data = np.expand_dims(image, axis=0)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    set_input_tensor(interpreter, input_data)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    #print(output_details)
    output = np.squeeze(interpreter.get_tensor(output_details['index']))

    # If the model is quantized (uint8 data), then dequantize the results
    if output_details['dtype'] == np.uint8:
        scale, zero_point = output_details['quantization']
        output = scale * (output - zero_point)

    #print("predection result")
    #print(output)

    return output

def checkPredictionResult(predictions):
    maxindex = np.argmax(predictions)
    direction = classname[maxindex]
    return direction

#########################追加部分(detect_obj()関数)###################
@smart_inference_mode()
def detect_object(
    img, # image
    weights=detect_modelfile,  # model path or triton URL
    data="root/src/rasp_yolov5/data/coco128.yaml",  # dataset.yaml path
    imgsz=(480, 640),  # inference size (height, width)
    conf_thres=0.15,  # 信頼度の閾値(ここは自分で調整しないと) 
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1,  # 一枚の画像でいくつの物体を検出するか(1000->1に変更)
    device="cpu",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
):
    label = None

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    with dt[0]:
        im = transforms.ToTensor()(img)
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

    # Inference
    with dt[1]:
        pred = model(im, augment=augment)
    # NMS
    with dt[2]:
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    # Process predictions
    for i, det in enumerate(pred):  # per image
        seen += 1
        im0 = img.copy()

        if len(det):
            # Rescale boxes from img_size to im0 size
            t = transforms.Compose([transforms.ToTensor()])
            im0  = t(im0);
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            y, x, w, h = int(det[0][0].item()), int(det[0][1].item()), int(det[0][2].item()), int(det[0][3].item())
            img = im.squeeze(0)
            img = img[:, x:h, y:w]
            # img = img[:, x:h, y:w].permute(1,2,0).numpy()
            # cv2.imshow('text', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # Print results
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()  # detections per class

            # Write results
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = names[c] if hide_conf else f"{names[c]}"
                confidence = float(conf)
                confidence_str = f"{confidence:.2f}"

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    #LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)

    return label

#########################################################################

#based on classify_picamera.py available from TensorFlow Lite Guide
def main():

    l = open(logfile, 'w')

    #invoking TFLite model
    t1 = time.perf_counter()
    interpreter = tflite.Interpreter(rccar_modelfile)
    interpreter.allocate_tensors()
    t2 = time.perf_counter()
    _, height, width, _ = interpreter.get_input_details()[0]['shape']
    #print(height, width)
    #l.write("model load time: " + str(round(t2 - t1, 6)) + '\n')

    i = 0
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    jpegfiles = []


    #for test of inferences
    if testimage == True:
        jpegfiles = glob.glob("test/left/*.jpeg") + glob.glob("test/right/*.jpeg") + glob.glob("test/straight/*.jpeg")

    current = 'straight'
    #main body
    try:
        while True:
            #print(time.perf_counter())
            t3 = time.perf_counter()
            #current time will be jpeg file name
            now = datetime.datetime.now()
            currenttime = '{0:%Y.%m.%d.%H%M%S%f}'.format(now)[:-4]

            #taking a picture or read a test file
            t4 = time.perf_counter()
            if testimage == False:
                jpegfile = getPicture(currenttime, cap)
            else:
                if i == len(jpegfiles):
                    break
                else:
                    jpegfile = jpegfiles[i]
                    i += 1
            t5 = time.perf_counter()

            #open the jpeg file
            l.write(jpegfile)
            l.write('\n')
            image_data = Image.open(jpegfile).resize((width, height))

            #######################追加(推論)########################
            t6 = time.perf_counter()
            detected_class = detect_object(image_data)
            t7 = time.perf_counter()
            if detected_class == "stop_obj":
                cap.release()
                fc.stop()
                l.close()
            #########################################################
            

            ##inference
            predictions = classify_image(interpreter, image_data)
            t8 = time.perf_counter()
            direction = checkPredictionResult(predictions)
            print('current', current, 'direction', direction, 'detected_class', detected_class)
            #l.write('current:' + str(current) + ' direction:' + str(direction) + 'detected_class:', detected_class, '\n')
            l.write('current:' + str(current) + ' direction:' + str(direction) + ' detected_class:' + str(detected_class) + '\n')
            t9 = time.perf_counter()

            #change direction
            if current == 'straight' and direction == 'straight':
                fc.forward(value[0])
                current = 'straight'
                dirname = imagedir + '/straight'
            elif current == 'straight' and direction == 'right':
                fc.turn_right(value[2], value[1])
                current = 'right'
                dirname = imagedir + '/right'
            elif current == 'straight' and direction == 'left':
                fc.turn_left(value[1], value[2])
                current = 'left'
                dirname = imagedir + '/left'                
            elif current == 'right' and direction == 'right':
                fc.turn_right(value[2], value[1])
                current = 'right'
                dirname = imagedir + '/right'
            elif current == 'right' and direction == 'straight':
                fc.forward(value[0])
                current = 'straight'
                dirname = imagedir + '/straight'
            elif current == 'right' and direction == 'left':
                fc.forward(value[0])
                current = 'straight'
                dirname = imagedir + '/straight'                
            elif current == 'left' and direction == 'left':
                fc.turn_left(value[1], value[2])
                current = 'left'
                dirname = imagedir + '/left'
            elif current == 'left' and direction == 'straight':
                fc.forward(value[0])
                current = 'straight'
                dirname = imagedir + '/straight'
            elif current == 'left' and direction == 'right':
                fc.forward(value[0])
                current = 'straight'
                dirname = imagedir + '/straight'                  
            #print("camera taking or image load time: " + str(round(t5 - t4, 6)))
            #print("inference time: " + str(round(t7 - t6, 6)))
            print("detection time:"+ str(round(t7-t6, 6)))
            print("classify time: " + str(round(t9 - t7, 6)))

            if testimage == False:
                cmd = ['mv', jpegfile, dirname]
            else:
                cmd = ['cp', jpegfile, dirname]
            subprocess.run(cmd)
            time.sleep(interval)
        #print("stop")
        cap.release()
        fc.stop()
        l.close()

            
    except KeyboardInterrupt:
        #print("stop")
        cap.release()
        fc.stop()
        l.close()
 
if __name__ == '__main__':
    main()
