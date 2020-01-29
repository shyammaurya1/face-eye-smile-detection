import numpy
import os 
import cv2 
import time
datasets = "faces"
size = 4
haar_file = 'haarcascade_frontalface_default.xml'
print('training...')

(images,lables,names,id)=([],[],{},0)

for (subdirs,dirs,files) in  os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        print(names[id])
        
        subjectpath = os.path.join(datasets,subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            lable = id
            images.append(cv2.imread(path,0)) #image is gray scale hence 0
            lables.append(int(lable))
        id +=1
(images,lables) = [numpy.array(lis) for  lis in [images,lables]]
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images,lables)
face_cascade = cv2.CascadeClassifier(haar_file)

(images,lables),names = (images,lables),names

camera = cv2.VideoCapture(0) #this make a web cam object
while True:
    retval, im = camera.read()
    gray = cv2.cvtColor(im , cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
        face =  gray[y:y+h,x:x+w]
        face_resize = cv2.resize(face,(130,100))
        prediction = model.predict(face_resize)
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),3)
        if prediction[1]<90:
            cv2.putText(im,'%s %.0f' %(names[prediction[0]],prediction[1]),(x-10,y-10),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0))
        else:
            cv2.putText(im,'Not Recognize',(x-10,y-10),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0))
            
    cv2.imshow('img',im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
camera.release()
cv2.destroyAllWindows()
import numpy
import os 
import cv2 
import time
datasets = "faces"
size = 4
haar_file = 'haarcascade_frontalface_default.xml'
print('training...')

(images,lables,names,id)=([],[],{},0)

for (subdirs,dirs,files) in  os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        print(names[id])
        
        subjectpath = os.path.join(datasets,subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            lable = id
            images.append(cv2.imread(path,0)) #image is gray scale hence 0
            lables.append(int(lable))
        id +=1
(images,lables) = [numpy.array(lis) for  lis in [images,lables]]
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images,lables)
face_cascade = cv2.CascadeClassifier(haar_file)

(images,lables),names = (images,lables),names

camera = cv2.VideoCapture(0) #this make a web cam object
while True:
    retval, im = camera.read()
    gray = cv2.cvtColor(im , cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
        face =  gray[y:y+h,x:x+w]
        face_resize = cv2.resize(face,(130,100))
        prediction = model.predict(face_resize)
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),3)
        if prediction[1]<90:
            cv2.putText(im,'%s %.0f' %(names[prediction[0]],prediction[1]),(x-10,y-10),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0))
        else:
            cv2.putText(im,'Not Recognize',(x-10,y-10),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0))
            
    cv2.imshow('img',im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
camera.release()
cv2.destroyAllWindows()
