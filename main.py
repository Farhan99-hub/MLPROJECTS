import streamlit as st
import tensorflow as tf
import numpy as np


# TensorFlow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("cance_cnn.h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element


# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Cancer Detection"])

# Main Page
if app_mode == "Home":
    st.header("CANCER DETECTION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Cancer Detection System! 🎗️🔬

    Our mission is to help in detecting cancer types efficiently. Upload a medical image, and our system will analyze it to detect any signs of cancer. Together, let's work towards better health outcomes!

    ### How It Works
    1. **Upload Image:** Go to the **Cancer Detection** page and upload an image related to cancer.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential cancer types.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for precise cancer detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, enabling quick medical decisions.

    ### Get Started
    Click on the **Cancer Detection** page in the sidebar to upload an image and experience the power of our Cancer Detection System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

# About Page
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset consists of medical images for cancer detection, categorized into various types. The dataset is used to train our model to recognize different cancer types.

    #### Content
    1. Training Images
    2. Validation Images
    3. Test Images
    """)

# Prediction Page
elif app_mode == "Cancer Detection":
    st.header("Cancer Detection")
    test_image = st.file_uploader("Choose an Image:")
    if test_image:
        st.image(test_image, width=400, use_column_width=True)

        if st.button("Predict"):
            st.snow()
            st.write("Our Prediction")
            result_index = model_prediction(test_image)
            # Reading Labels
            class_names = ['Acute Lymphoblastic Leukemia (ALL) - Benign Cases', 'Acute Lymphoblastic Leukemia (ALL) - Early Stage', 'Acute Lymphoblastic Leukemia (ALL) - Pre-Treatment', 'Acute Lymphoblastic Leukemia (ALL) - Progressed Cases', 'Brain Cancer - General Tumor', 'Brain Cancer - Glioma', 'Brain Cancer - Meningioma', 'Breast Cancer - Benign', 'Breast Cancer - Malignant', 'Cervical Cancer - Abnormal Pap Results', 'Cervical Cancer - Dyskaryosis', 'Cervical Cancer - Keratinized Cells', 'Cervical Cancer - Pre-Cancerous Cells', 'Cervical Cancer - Specific Forms', 'Kidney Cancer - Normal Tissue', 'Kidney Cancer - Tumor', 'Lung and Colon Cancer - Benign Colon Conditions', 'Lung and Colon Cancer - Benign Lung Conditions', 'Lung and Colon Cancer - Colon Adenocarcinoma', 'Lung and Colon Cancer - Lung Adenocarcinoma', 'Lung and Colon Cancer - Lung Squamous Cell Carcinoma', 'Lymphoma - Chronic Lymphocytic Leukemia', 'Lymphoma - Follicular Lymphoma', 'Lymphoma - Mantle Cell Lymphoma', 'Oral Cancer - Normal Tissue', 'Oral Cancer - Squamous Cell Carcinoma']
            st.success(f"Model predicts: {class_names[result_index]}")

# Add footer with trademark
st.markdown("""
<style>
footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background-color: #f1f1f1;
    padding: 5px;
    text-align: center;
    font-size: 12px;
    color: #555;
}
</style>
<footer>
    <p>© 2024 Farhan Sana Ansari. All rights reserved.</p>
</footer>
""", unsafe_allow_html=True)