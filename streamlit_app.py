import os

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt


if __name__ == '__main__':
    model = load_model('model/base_model_no_tl.h5')

    explanation_expander = st.beta_expander('Some explanation on lesions')
    with explanation_expander:
        st.write('try this')

    st.title('Skin cancer detector')


    uploaded_file = st.file_uploader('Upload an lesion image')
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        # plot
        # fig, ax = plt.subplots()
        # plt.imshow(img)
        # st.pyplot(fig)

        def prepare_image(img):
            tf_image = np.array(img)
            img_resized = np.resize(tf_image, (224,224,3))
            st.write(img_resized.shape)
            img_resized = img_resized[:, :, ::-1]       # RGB to BGR
            img_reshaped = img_resized.reshape((1, img_resized.shape[0], img_resized.shape[1], img_resized.shape[2]))
            st.write(img_reshaped.shape)
            img_scaled = img_reshaped/255
            st.write(img_scaled)

        prepared_image = prepare_image(img)


    # pred_array = model.predict(prepared_image)
    # st.write(pred_array)
    # st.write('is this an index?', np.argmax(pred_array))

    cancerous_classes = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc', 'mel']

    st.write('This is the predicted class.')
    # if predicted_class in cancerous_classes:
    #     st.write('Bad news')
    # else:
    #     st.write('Good news')


    hide_st_style = """ <style> footer {visibility: hidden;} </style> """
    st.markdown(hide_st_style, unsafe_allow_html=True)





    def print_score(prediction: list):
        pred = prediction[0][0]
        if pred > 0.7:
            print(f"{round(pred * 100, 4)}%")
        elif pred < 0.3:
            print(f"{round(pred * 100, 4)}%")
        else:
            print(f"{round(pred * 100, 4)}%")


    # prediction = model.predict(prepare_image(path))
    # print(prediction)


