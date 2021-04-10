#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries

# In[1]:


from matplotlib import pyplot

#from mtcnn.mtcnn import MTCNN
from matplotlib.patches import Rectangle
from keras.models import load_model
from matplotlib.patches import Circle
import cv2
from facenet_pytorch import MTCNN
import numpy as np
from sklearn.preprocessing import Normalizer
from scipy.spatial.distance import cosine
import os
import datetime
import xlwt
import xlrd
from xlwt import Workbook 
from xlutils.copy import copy
import time
# from facenet_pytorch import MTCNN
# from cv2 import dnn_superres


# ### Load model of MTCNN and FaceNet

# In[2]:


encoder_model = 'facenet_keras.h5'


# In[ ]:





# In[3]:


from platform import python_version

print(python_version())


# In[4]:


detector=MTCNN()


# In[ ]:


face_encoder = load_model(encoder_model)
people_dir = './people'
encoding_dict = dict()


# In[ ]:


classNames = []
with open('coco.names','r') as f:
    classNames = f.read().splitlines()
print(classNames)


# In[ ]:


weightsPath = "frozen_inference_graph.pb"
configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


# In[ ]:


# sr = cv2.dnn_superres.DnnSuperResImpl_create()
# path = "LapSRN_x8.pb"
# sr.readModel(path)

# # Set the desired model and scale to get correct pre- and post-processing
# sr.setModel("lapsrn", 8)

# sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


# ### Encoding of face using FaceNet

# In[ ]:


def get_encode(face_encoder, face, size):
    face = normalize(face)
    face = cv2.resize(face, size)
    encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
    return encode


# ### Extract Face for encoding

# In[ ]:


def get_face(img, box):
    [[x1, y1, width, height]] = box
    x1, y1 ,x2,y2= int(x1), int(y1),int(width),int(height)
    #x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)


# In[ ]:


"""def get_face(img, box):
    [[x1, y1, width, height]] = box
    x1, y1 ,x2,y2= int(x1), int(y1),int(width),int(height)
    #x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)"""


# In[ ]:


def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std


# In[ ]:


l2_normalizer = Normalizer('l2')


# In[ ]:


def export(a):
    ct = datetime.datetime.now()
    
    ws = xlrd.open_workbook('names_list.xls',formatting_info=True)
    sheet = ws.sheet_by_index(0)
    wb=copy(ws)
    names_sheet=wb.get_sheet(0)
    row_s=sheet.nrows
    col_s=sheet.ncols
    
    names_sheet.write(0,col_s,str(ct.year)+"_"+str(ct.month)+"_"+str(ct.day)+"_"+str(ct.hour))
    
    for i in range(1,row_s):   
        
        if sheet.cell_value(i,0) in a:
            names_sheet.write(i, col_s,"P")
                
        else:
                
            names_sheet.write(i, col_s,"A")
    wb.save("names_list.xls") 


# In[ ]:


def mark_attendance(a):
    

    ct = datetime.datetime.now()
    
    


    workbook = xlwt.Workbook()  

    sheet = workbook.add_sheet(str(ct.year)+"_"+str(ct.month)+"_"+str(ct.day)) 
    sheet.write(0,0,"Name")


    row = 1
    col = 0
    if len(a)>0:
        for person_name in os.listdir(people_dir):
            print(person_name)
            print(a)

            for x in range(0,len(a)):
                if str(person_name) in a:
                    sheet.write(row, col,     str(a[x]))
                    sheet.write(row,col+1,"P")
                    
                else:
                    sheet.write(row, col,     str(a[x]),)
                    sheet.write(row,col+1,"A")
                row+=1
         
            
  
        workbook.save("sample_class_1.xls") 
    else:
        sheet.write(1,0,"No one is present")
        workbook.save("sample_class_1.xls") 


# ### Saved Images whhich are encoded and stored in a Dictionary

# In[ ]:


for person_name in os.listdir(people_dir):
    person_dir = os.path.join(people_dir, person_name)
    encodes = []
    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        img = cv2.imread(img_path)
        boxes,probs = detector.detect(img)

        #results = detector.detect_faces(img)
        if boxes is not None:
            print(boxes)
        
            #res = max(results, key=lambda b: b['box'][2] * b['box'][3])
            face, _, _ = get_face(img, boxes.tolist())
            
            face = normalize(face)
            face = cv2.resize(face,(160,160))
            encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
            encodes.append(encode)
    if encodes:
        encode = np.sum(encodes, axis=0)
        encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]
        encoding_dict[person_name] = encode
 


# In[ ]:


recognition_t=0.6
confidence_t=0.99
thres = 0.5 # Threshold to detect object
nms_threshold = 0.2 #(0.1 to 1) 1 means no suppress , 0.1 means high suppress
font = cv2.FONT_HERSHEY_PLAIN
#font = cv2.FONT_HERSHEY_COMPLEX
Colors = np.random.uniform(0, 255, size=(len(classNames), 3))


# ### Live Face Detection and Recognition (Press q to end video stream)

# In[ ]:


from matplotlib import pyplot as plt


# RTSP 

# In[ ]:


"""rtsp_username = "admin"
rtsp_password = "admin@123"
##change the channel no according to the one out of 4
rtsp = "rtsp://" + rtsp_username + ":" + rtsp_password + "@192.168.11.110:554/Streaming/channels/" + channel + "02" #change the IP to suit yours
cap = cv2.VideoCapture()
cap.open(rtsp)"""


# In[ ]:


#video =cv2.VideoCapture('rtsp://admin:admin@192.168.1.240/1') 
#https://stackoverflow.com/questions/49978705/access-ip-camera-in-python-opencv


# In[ ]:


import urllib.request
url='http://192.168.1.100:8080/shot.jpg?rnd=927909'

    
    


# In[ ]:


#video =cv2.VideoCapture('http://192.168.1.100:8080/')
present_candidates=[]
fps_start_time = datetime.datetime.now()
fps = 0
total_frames = 0

#frame=cv2.imread("actual_frame_2.png")
#lt.show(frame)
sr = cv2.dnn_superres.DnnSuperResImpl_create()
path = "ESPCN_x2.pb"
#path = "LapSRN_x2.pb"
sr.readModel(path)

# Set the desired model and scale to get correct pre- and post-processing
sr.setModel("espcn", 2)
#sr.setModel("lapsrn", 2)

#d = IPython.display.display("", display_id=1)

while True:
    imgResp=urllib.request.urlopen(url)
    imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
    frame=cv2.imdecode(imgNp,-1)
    #check,frame=video.read()
    #frame=sr.upsample(frame)
    total_frames = total_frames + 1

    faces,_=detector.detect(frame)

    if faces is not None:
        for person in faces:
            #print(person)
            bounding_box=person
            #keypoints=person["keypoints"]
    #             if person['confidence'] < confidence_t:
    #                 continue
            face, pt_1, pt_2 = get_face(frame, [bounding_box])
            encode = get_encode(face_encoder, face,(160,160))
            encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
            name = 'unknown'


            distance = float("inf")


            for (db_name, db_enc) in encoding_dict.items():

                dist = cosine(db_enc, encode)

                if dist < recognition_t and dist < distance:

                    name = db_name
                    distance = dist
                    if name not in present_candidates:
                        present_candidates.append(name)


            if name == 'unknown':
                cv2.rectangle(frame, pt_1, pt_2, (0, 0, 255), 2)
                cv2.putText(frame,name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

            else:
                cv2.rectangle(frame, pt_1, pt_2, (0, 255, 0), 2)
                cv2.putText(frame,name + f'__{distance:.2f}', (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 200, 200), 2)

    fps_end_time = datetime.datetime.now()
    time_diff = fps_end_time - fps_start_time
    if time_diff.seconds == 0:
        fps = 0.0
    else:
        fps = (total_frames / time_diff.seconds)

    fps_text = "FPS: {:.2f}".format(fps)

    cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

    cv2.imshow('n',frame) 
    #print("check")
    #cv2.waitKey(0)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        
        break
            
            
            
video.release()

cv2.destroyAllWindows()


# In[ ]:


count=0
for no,obj in enumerate(classIds):
    if obj==[1]:
        count+=1
        
        
print(count)#####Number of people detected 


# ### Snap based Face Detection and recognition (Press Spacebar to take a snap and Escape to end the stream)

# In[ ]:


"""video1 =cv2.VideoCapture('rtsp://admin:admin@192.168.1.240/1') 
#https://stackoverflow.com/questions/49978705/access-ip-camera-in-python-opencv"""


# In[ ]:


"""
present_candidates_1=[]
img_counter=0
fps_start_time = datetime.datetime.now()
fps = 0
total_frames = 0
while True:
    
    check,frame=video1.read()
    total_frames = total_frames + 1
    frame=sr.upsample(frame)
    
    fps_end_time = datetime.datetime.now()
    time_diff = fps_end_time - fps_start_time
    if time_diff.seconds == 0:
        fps = 0.0
    else:
        fps = (total_frames / time_diff.seconds)

    fps_text = "FPS: {:.2f}".format(fps)

    cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
    
    cv2.imshow("SS",frame)
    k=cv2.waitKey(1)
    
    if k%256 == 27:
        print("Escape")
        break
        
    elif k%256==32:
        faces=detector.detect_faces(frame)
        if faces !=[]:
            for person in faces:
                bounding_box=person["box"]
                keypoints=person["keypoints"]
    #             if person['confidence'] < confidence_t:
    #                 continue
                face, pt_1, pt_2 = get_face(frame, person['box'])
                encode = get_encode(face_encoder, face,(160,160))
                encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
                name = 'unknown'


                distance = float("inf")


                for (db_name, db_enc) in encoding_dict.items():

                    dist = cosine(db_enc, encode)

                    if dist < recognition_t and dist < distance:

                        name = db_name
                        distance = dist
                        if name not in present_candidates_1:
                            present_candidates_1.append(name)

                if name == 'unknown':
                    cv2.rectangle(frame, pt_1, pt_2, (0, 0, 255), 2)
                    cv2.putText(frame,name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

                else:
                    cv2.rectangle(frame, pt_1, pt_2, (0, 255, 0), 2)
                    cv2.putText(frame,name + f'__{distance:.2f}', (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 200, 200), 2)
                    
        
                    
        img_name="opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name,frame)
        print("Screenshot")
        img_counter+=1

video1.release()
cv2.destroyAllWindows()"""


# ### Candidates present (Live video)

# In[ ]:


present_candidates


# ### Candidates present (Snap)

# In[ ]:


#present_candidates_1


# In[ ]:


#export(present_candidates_1)


# In[ ]:


export(present_candidates)


# In[ ]:


len(faces)#Number of faces detected


# In[ ]:




