import streamlit as st
from PIL import Image
import cv2
import numpy as np
st.title("Face detection")
upload_file = st.file_uploader('upload image',type=['jpg','png',"jpeg"])
if upload_file is not None:
    image = Image.open(upload_file)
    np_image = np.array(image)
    image_bgr = cv2.cvtColor(np_image,cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_bgr,cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(gray,1.1,4)#-->#scalefactor # minneighbour

    for (x,y,w,h) in faces:
     cv2.rectangle(image_bgr,(x,y),(x+w,y+h),(0,255,0),2)

    result_rgb = cv2.cvtColor(image_bgr,cv2.COLOR_BGR2RGB)
    st.image(result_rgb,caption='Face Detection',use_column_width=True)