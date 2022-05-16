import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import streamlit as st
import easyocr as ocr
from PIL import Image
import scan as s
import numpy as np
import qr_code_app as qr
import cv2

st.title('Document/QR code Scanner')
option = st.selectbox(
     'Would you like to scan a document or QR code?',
     ('Document','QR code','Text Detection'))

st.write('Upload the image to scan your:', option)
uploaded = st.file_uploader("Upload your picture", type=("png", "jpg","jpeg"))
if uploaded is not None:
	st.write('Sucessfully Uploaded!')
	if option == 'Document':
		img = Image.open(uploaded)
		st.image(img,width=400)
		image=np.array(img)
		st.write('Detecting edges...')
		edged, ratio=s.edge_detection(image)
		egde_pic=Image.fromarray(edged)
		st.image(egde_pic)
		st.write('Finding countour...')
		contoured,screenCnt=s.find_contours(image,edged)
		contoured_pic=Image.fromarray(contoured)
		st.image(contoured_pic)
		st.write('Almost Ready...')
		final=s.perspective(image,screenCnt,ratio)
		final_pic=Image.fromarray(final)
		st.image(final_pic)
		if st.button("Download"):
			s.download(final)
			st.write('Downloaded Sucessfully')

	if option == 'QR code':
		image = np.array(Image.open(uploaded))

		st.subheader('Original Image')

		#display the image
		st.image(image, caption=f"Original Image", use_column_width=True)

		st.subheader('Decoded data')

		decode_data = qr.qr_code_dec(image)
		st.markdown(decode_data)


	@st.cache
	def load_model():
		reader = ocr.Reader(['en'], model_storage_directory='.')
		return reader

	reader = load_model()  # load model

	if option == 'Text Detection':

		input_image = Image.open(uploaded)  # read image
		st.image(input_image)  # display image

		with st.spinner("🤖 AI is at Work! "):
			result = reader.readtext(np.array(input_image))
			result_text = []  # empty list for results

			for text in result:
				result_text.append(text[1])

			st.write(result_text)
		st.balloons()
	else:
		st.write("Upload an Image")
