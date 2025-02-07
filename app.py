# Python In-built packages
from pathlib import Path
import PIL
#https://www.youtube.com/watch?v=UaHRkS7d8Ks
#https://www.youtube.com/watch?v=LmNMLhMRZKE&t=6s
# External packages
import streamlit as st

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Object Detection",
    page_icon="./images/logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Нейронная сеть для детектирования заболеваний растений")

# Sidebar
st.sidebar.image(image = "./images/logo.png", width=200)
st.sidebar.header("Urpaq-bio AI")


# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Детектирование', 'Сегментация (старт проекта март 2025)'])

confidence = float(st.sidebar.slider(
    "Выберите точность распознавания (лучше 25)", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == 'Детектирование':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Сегментация (старт проекта март 2025)':
    model_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Невозможно загрузить изображение, проверьте папку с фалами: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Выберите источник", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Выберите картинку растения...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None  :
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                print(boxes)
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!")

elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

elif source_radio == settings.RTSP:
    helper.play_rtsp_stream(confidence, model)

elif source_radio == settings.YOUTUBE:
    helper.play_youtube_video(confidence, model)

else:
    st.error("Выберите подходящий источник видео!")

st.write('Руководитель кружка "Флорариум әлемінде": ПДО Абраева Инеш Бахытжановна')
st.write('Автор проекта: Газиз Темирлан')