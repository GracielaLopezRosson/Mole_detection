import streamlit as st
from tensorflow.keras.models import load_model

if __name__ == '__main__':
    st.title('Skin cancer detector')
    uploaded_file = st.file_uploader('Upload an lesion image')

    explanation_expander = st.beta_expander('jfjdsfgksn')
    with explanation_expander:
        st.write('try this')

    # model = load_model('model.h5')

    # predicted_class = model.predict

    cancerous_classes = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc', 'mel']

    st.write('This is the predicted class.')
    # if predicted_class in cancerous_classes:
    #     st.write('Bad news')
    # else:
    #     st.write('Good news')


    hide_st_style = """ <style> footer {visibility: hidden;} </style> """
    st.markdown(hide_st_style, unsafe_allow_html=True)