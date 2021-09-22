import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt

if __name__ == '__main__':
    model = load_model('model/base_model_no_tl.h5')

    st.title('Skin cancer detector')
    uploaded_file = st.file_uploader('Upload an lesion image')
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        fig, ax = plt.subplots()
        plt.imshow(img)
        st.pyplot(fig)
    explanation_expander = st.beta_expander('Some explanation on lesions')
    with explanation_expander:
        st.write('try this')


    # predicted_class = model.predict

    cancerous_classes = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc', 'mel']

    st.write('This is the predicted class.')
    # if predicted_class in cancerous_classes:
    #     st.write('Bad news')
    # else:
    #     st.write('Good news')


    hide_st_style = """ <style> footer {visibility: hidden;} </style> """
    st.markdown(hide_st_style, unsafe_allow_html=True)


    def prepare_image(image_path: str):
        img = image.load_img(image_path, target_size=(224, 224))
        img = image.img_to_array(img)
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))

        # we preprocess ourselves ?
        preprocessed_image = None
        return preprocessed_image


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


