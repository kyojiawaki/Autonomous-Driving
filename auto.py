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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import re
import sys
import cv2
import time
import datetime
import glob
import numpy as np
import tflite_runtime.interpreter as tflite # TF2
from six.moves import urllib
from PIL import Image, ImageFilter, ImageChops
from operator import itemgetter
import RPi.GPIO as GPIO
import subprocess
import picar_4wd as fc

#tflite setup
input_mean = 127.5
input_std = 127.5
#location of model file
#modelfile = "awaki_model/rccar/converted_model_resnet.tflite"
#modelfile = "awaki_model/rccar/converted_model_mid_deep.tflite"
#modelfile ="awaki_model/rccar/resnet.tflite"
modelfile="awaki_model/rccar/inc_mid.tflite"
classname = ['left', 'right', 'straight']

#output files
name = sys.argv[1]
imagedir = "image/" + name
logfile = "log/" + name + ".log"

testimage = False

#change speed for straight (value[0]) or left/right (value[1] and value[2])
#value = [30, 8, 70]
value = [1,0, 30] 
interval = 0.1

if not os.path.exists(imagedir):
    os.makedirs(imagedir)
    dirname = imagedir + '/left'
    os.makedirs(dirname)
    dirname = imagedir + '/right'
    os.makedirs(dirname)
    dirname = imagedir + '/straight'
if not os.path.exists('log'):
    os.makedirs('log')

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

#based on classify_picamera.py available from TensorFlow Lite Guide
def main():

    l = open(logfile, 'w')

    #invoking TFLite model
    t1 = time.perf_counter()
    interpreter = tflite.Interpreter(modelfile)
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
           # print(time.perf_counter())
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
            #inference
            t6 = time.perf_counter()
            predictions = classify_image(interpreter, image_data)
            t7 = time.perf_counter()
            direction = checkPredictionResult(predictions)
            print('current', current, 'direction', direction)
            l.write('current:' + str(current) + ' direction:' + str(direction) + '\n')
            t8 = time.perf_counter()

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
            #print("one operation time: " + str(round(t8 - t3, 6)))

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
