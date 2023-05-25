# It's training now........
# Ok~ in this video, I'll show you how to train your customized labeled data.
# In this use case, I'll train to detect the car lisence plate. And then use EasyOCR to recognize character.

# About how to prepare our data to train your model, here I mainly download pics from "flickr".
# I totally download 17 pics (all are with very high resulution) here you see~
# And each pic should be tackled/labeled by LabelImg like this~
# With LabelImg program,  you must convert to "YOLO" mode, not "PascalPOC" mode~
# After you label, you will see the result txt file like this~
# And you must name the txt corresponding to the png, they should be the "same" name~

# In this tutorial, it's just for a test, so if you want to train well, you must add more pics....
# And in this use case, I also use EASYOCR here~
# You can know I have two part in the case:
#   1) Train model.
#   2) Detect Character in bounding box with model prediction. EasyOCR area is here~



##  Train your YOLO (As you see, with ultralytics we just run code in python environment)
# And in fact, you can use SOAT model called Yolov8 as well
# Let's check the yaml file~~~~  it's also very very simple!!!!!
# Why we set here, it's because its default dataset file setting, so you don't need to set like this~

from ultralytics import YOLO
model   = YOLO("yolov5s6.pt")                           # I try many many models from ultralytics, however, I think this one suits for me best.
results = model.train(data="config.yaml", epochs=15)    # But if you want to train yolov8 type model, it's fine as well.

# # check model prediction first  seems perfect
# results = model(r'C:\Users\vito\Desktop\python_yolov8_train\datasets\train\images\3.png')
# boxes   = results[0].boxes            ;       dir(results[0])
# box     = boxes[0]  # returns one box
# box.xyxy





## Do some analysis
# OK~ Let's check the main part



import  cv2
import  easyocr
from    PIL                 import Image
import  numpy               as np
import  matplotlib.pyplot   as plt


reader      = easyocr.Reader(['en'])

def score_frame(frame):                                                                     # :param frame: input frame in numpy/list/tuple format.                              
    frame   = [frame]                                                                       # Takes a single frame as input, and scores the frame using yolo5 model.
    results = model(frame)                                                                  # :return: Labels and Coordinates of objects detected by model in the frame.
    results = results[0].boxes
    labels  = results.cls                                                                   # results.pandas().xyxy[0]
    cord    = results.xyxyn
    conf    = results.conf
    return labels, cord, conf                                               



def plot_boxes(results, frame):
    labels, cord, conf      = results                                                       # Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
    n                       = len(labels)                                                   # :param results: contains labels and coordinates predicted by model on the given frame.
    x_shape, y_shape        = frame.shape[1], frame.shape[0]                                # :return: Frame with bounding boxes and labels ploted on it.
    for i in range(n):                                                                      # i=0
        row                 = cord[i]
        if conf >= 0.2:
            x1, y1, x2, y2  = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
            cv2.rectangle(
                img         = frame, 
                pt1         = (x1, y1),                                                     # part1 Vertex of the rectangle
                pt2         = (x2, y2),                                                     # part2 Vertex of the rectangle opposite to pt1
                color       = (0, 255, 0) if model.names[int(labels[i])]!='person' else (0, 0, 255), 
                thickness   = 2
            )
            ####################### for EASYOCR ###############################
            plate_region    = frame[y1:y2, x1:x2]
            plate_image     = Image.fromarray(cv2.cvtColor(plate_region, cv2.COLOR_BGR2RGB))
            plate_array     = np.array(plate_image)
            plate_number    = reader.readtext(plate_array)
            concat_number   = ' '.join([number[1] for number in plate_number])
            number_conf     = np.mean([number[2] for number in plate_number])
            #####################################################################
            cv2.putText(
                img         = frame, 
                text        = concat_number+f"(Conf:{round(number_conf,2)})",               # text = model.names[int(labels[i])]+f":{round(float(conf),1)}", 
                org         = (x1, y1),                                                     # Bottom-left corner of the text string in the image.
                fontFace    = cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale   = 0.8, 
                color       = (0, 255, 0) if model.names[int(labels[i])]!='person' else (0, 0, 255), 
                thickness   = 1
            )
    return frame



image_path  = r'C:\Users\vito\Desktop\python_yolov8_train\datasets\train\images\2.png'
image_path  = r'C:\Users\vito\Desktop\python_yolov8_train\5257.jpg'
image       = Image.open(image_path)
frame       = np.array(image)
results     = score_frame(frame)                                    
frame       = plot_boxes(results, frame)            # plt.imshow(frame) ; plt.show()



# OK~ Seems great, see ya!!!!!!!!!!