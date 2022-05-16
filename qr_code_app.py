import streamlit as st

from PIL import Image
import numpy as np
import cv2


#title of the web-app
st.title('QR Code Decoding with OpenCV')

@st.cache
def show_qr_detection(img,pts):

    pts = np.int32(pts).reshape(-1, 2)

    for j in range(pts.shape[0]):

        cv2.line(img, tuple(pts[j]), tuple(pts[(j + 1) % pts.shape[0]]), (255, 0, 0), 5)

    for j in range(pts.shape[0]):
        cv2.circle(img, tuple(pts[j]), 10, (255, 0, 255), -1)


@st.cache
def qr_code_dec(image2):

    decoder = cv2.QRCodeDetector()

    data, vertices, rectified_qr_code = decoder.detectAndDecode(image2)
    if len(data) > 0:
        print("Decoded Data: '{}'".format(data))

    # Show the detection in the image:
        show_qr_detection(image2, vertices)

        rectified_image = np.uint8(rectified_qr_code)

        decodes_data = 'Decoded data: '+ data

        rectified_image = cv2.putText(rectified_image,decodes_data,(50,350),fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale = 2,
            color = (250,225,100),thickness =  3, lineType=cv2.LINE_AA)


        return decodes_data