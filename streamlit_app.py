import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

from utils import data_preprocessing as dp

if __name__ == '__main__':

    model = load_model('model/v3_mobilenetv2.h5')

    st.set_page_config(page_title="skin cancer predictor", layout="wide")
    hide_st_style = """ <style> footer {visibility: hidden;} </style> """

    explanation_expander = st.beta_expander('Lesion legend')
    with explanation_expander:
        st.markdown("<h3 style='text-align: left; color: #deb887; font-size:15px;'>* akiec: Actinic keratoses and intraepithelial carcinoma "
                    "(Bowen\'s disease)</h3>",
                    unsafe_allow_html=True)
        st.markdown(
            "<h3 style='text-align: left; color:grey; font-size:13px; margin-left:30px'>Variants of squamous cell"
            " carcinoma that can be treated locally without surgery,"
            "and commonly non-invasive", unsafe_allow_html=True)

        st.markdown(
            "<h3 style='text-align: left; color: #deb887; font-size:15px;'>* bcc: basal cell carcinoma",
            unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: left;color:grey; font-size:13px; margin-left:30px'>A common variant of epithelial skin"
                    " cancer that rarely metastasizes but grows destructively if untreated)", unsafe_allow_html=True)

        st.markdown(
            "<h3 style='text-align: left; color: #deb887; font-size:15px;'>* bkl: benign keratosis-like lesions",
            unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: left;color:grey; font-size:13px; margin-left:30px'"
                    ">They are similar biologically and often reported under the same generic term histopathologically",
                    unsafe_allow_html=True)

        st.markdown(
            "<h3 style='text-align: left; color: #deb887; font-size:15px;'>* df: dermatofibroma ",
            unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: left;color:grey; font-size:13px; margin-left:30px'"
                    ">Benign skin lesion regarded as either a benign proliferation or an inflammatory reaction"
                    " to minimal trauma",
                    unsafe_allow_html=True)

        st.markdown(
            "<h3 style='text-align: left; color:grey; color: #deb887; font-size:15px;'>* nv: melanocytic nevi  ",
            unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: left; color: grey; font-size:13px; margin-left:30px'"
                    ">Benign neoplasms of melanocytes, which appear in many variants",
                    unsafe_allow_html=True)

        st.markdown(
            "<h3 style='text-align: left; color:grey; color: #deb887; font-size:15px;'>* vasc: vascular lesions  ",
            unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: left; color: grey; font-size:13px; margin-left:30px'"
                    ">Vascular lesions, which are pigmented by hemoglobin and not by melanin",
                    unsafe_allow_html=True)


        st.markdown(
            "<h3 style='text-align: left; color: #deb887; font-size:15px;'>* mel: melanoma  ",
            unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: left;color:grey; color: grey; font-size:13px; margin-left:30px'"
                    ">The deadliest type of skin cancer. It  is a malignant neoplasm derived"
                    " from melanocytes that may appear in different variants",
                    unsafe_allow_html=True)


    st.title('Skin cancer detector')

    uploaded_file = st.file_uploader('Upload an lesion image')
    if uploaded_file is not None:
        # valid_file = False
        try:
            img = Image.open(uploaded_file)
            # valid_file = True

            classes = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc', 'mel']
            cancerous_classes = ['mel', 'bkl', 'bcc']  # bkl = warning

            prepared_image = dp.prepare_one_image_no_tl(img)
            pred_array = model.predict(prepared_image)

            pred_array_df = pd.DataFrame(pred_array, columns=classes)
            st.write(pred_array_df)

            pred_index = np.argmax(pred_array)
            predicted_class = classes[pred_index]
            st.write('The predicted lesion type:', predicted_class)
            if predicted_class in cancerous_classes:
                st.write('Diagnosis: cancerous')
            else:
                st.write('Diagnosis: not cancerous')


        except:
            st.write('Sorry but we do not understand the uploaded file.'
                     ' Please make sure to upload an image file.')

    hide_st_style = """ <style> footer {visibility: hidden;} </style> """
    st.markdown(hide_st_style, unsafe_allow_html=True)
