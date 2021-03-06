# -*- coding: utf-8 -*-
"""Detection+Recognition.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1cfvfeMbcMJlLFsgfGDb-Bg6-Af-cqN_y

### Import Libraries
"""


import datetime
import xlwt
import xlrd
from xlwt import Workbook 
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
from matplotlib.patches import Rectangle
from keras.models import load_model
from matplotlib.patches import Circle
import cv2
import numpy as np
from sklearn.preprocessing import Normalizer
from scipy.spatial.distance import cosine
import os
from xlutils.copy import copy
"""### Load model of MTCNN and FaceNet"""

encoder_model = 'model/facenet_keras.h5'

detector=MTCNN()
face_encoder = load_model(encoder_model)
people_dir = 'photos'
encoding_dict = dict()

"""### Encoding of face using FaceNet"""

def get_encode(face_encoder, face, size):
    face = normalize(face)
    face = cv2.resize(face, size)
    encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
    return encode

"""### Extract Face for encoding"""

def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)

def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

l2_normalizer = Normalizer('l2')
### collect daywise attendance by checking through a list of ppl
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
### enter names of ppl present   
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

"""### Saved Images whhich are encoded and stored in a Dictionary"""

for person_name in os.listdir(people_dir):
    person_dir = os.path.join(people_dir, person_name)
    encodes = []
    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        img = cv2.imread(img_path)

        results = detector.detect_faces(img)
        if results:
            res = max(results, key=lambda b: b['box'][2] * b['box'][3])
            face, _, _ = get_face(img, res['box'])
            
            face = normalize(face)
            face = cv2.resize(face,(160,160))
            encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
            encodes.append(encode)
    if encodes:
        encode = np.sum(encodes, axis=0)
        encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]
        encoding_dict[person_name] = encode

recognition_t=0.6
confidence_t=0.99

"""### Live Face Detection and Recognition (Press q to end video stream)"""

video =cv2.VideoCapture(0)
present_candidates=[]

while True:
    check,frame=video.read()

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
                    if name not in present_candidates:
                        present_candidates.append(name)
                    
                    
            if name == 'unknown':
                cv2.rectangle(frame, pt_1, pt_2, (0, 0, 255), 2)
                cv2.putText(frame,name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                
            else:
                cv2.rectangle(frame, pt_1, pt_2, (0, 255, 0), 2)
                cv2.putText(frame,name + f'__{distance:.2f}', (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 200, 200), 2)
                

       
    cv2.imshow('frame',frame)  
    if cv2.waitKey(5) & 0xFF == ord('q'):
          break
            
            
            
video.release()
cv2.destroyAllWindows()

"""### Snap based Face Detection and recognition (Press Spacebar to take a snap and Escape to end the stream)"""

video1 =cv2.VideoCapture(0)
present_candidates_1=[]
img_counter=0
while True:
    
    check,frame=video1.read()
    
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
cv2.destroyAllWindows()

"""### Candidates present (Live video)"""

export(present_candidates_1)

"""### Candidates present (Snap)"""

export(present_candidates)


