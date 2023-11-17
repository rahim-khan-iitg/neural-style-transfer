import streamlit as st
from model import apply_style
st.title("Neural Style Transfer")




with st.spinner("Loading"):
    content_image=st.file_uploader("Upload Content Image",type=['jpg','jpeg','png'])
    if content_image is not None:
        img=content_image.getvalue()
        st.image(img)

with st.spinner("Loading"):
    style_image=st.file_uploader("Upload Style Image",type=['jpg','jpeg','png'])

if style_image is not None:
        img=style_image.getvalue()
        st.image(img)
        # make(content_image,style_image)
iterations=st.number_input(label="number of iterations ", min_value=0,max_value=2500,value=500)
if content_image is not None and style_image is not None:
    st.image(apply_style(content_image,style_image,20))