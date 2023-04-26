# Import necessary libraries
import streamlit as st
from PIL import Image
import numpy as np
from io import BytesIO
import os
import torch
import torchvision.transforms as transforms
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import easyocr

# Load the pre-trained models and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Set the device to be used for inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Text extraction from image
def get_text_from_image(image_path):
    # Load OCR model
    reader = easyocr.Reader(['en'])

    # # Load image and extract text
    text = reader.readtext(image_path)

    # print(text)
    # Preprocess text
    preprocessed_text = " ".join([t[1] for t in text])

    return preprocessed_text


# Function to perform image captioning
def get_caption(image_paths, num_return_sequences):
    max_length = 16
    num_beams = 7
    # num_return_sequences = 5  # number of captions to generate for each image
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "num_return_sequences": num_return_sequences}
    captions_list = []
    for image_path in image_paths:
        # Load the image from the image path
        i_image = Image.fromarray(image_path)

        # Convert the image to RGB format if not already in RGB format
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        # Preprocess the image and convert it to tensor format
        pixel_values = feature_extractor(images=[i_image], return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        ocr_text = get_text_from_image(image_path)
        if ocr_text:
            captions_list.append(ocr_text)
            gen_kwargs["num_return_sequences"] = num_return_sequences - 1

        # print('before',captions_list)
        # Generate the caption for the image using the pre-trained model and tokenizer
        output_ids = model.generate(pixel_values, **gen_kwargs)
        captions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        # Append the captions for the image to the captions list
        # print('captions',captions)
        captions_list += captions
        # print('after',captions_list)

    return captions_list


# Streamlit app
st.title(" Caption Generator")
num_return_sequences = st.selectbox("Select the number of captions you want to generate:", options=[1, 2, 3, 4, 5, 6],index=3)

# Allow user to upload images
uploaded_files = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
print('uploaded_file', uploaded_files)
if uploaded_files is not None:
    image_paths = []
    for uploaded_file in uploaded_files:
        # Read the image data from the file object
        image_data = uploaded_file.getvalue()

        # Convert the image data to a PIL Image object
        image = Image.open(BytesIO(image_data))

        # Convert the image to a numpy array
        image_array = np.array(image)

        # Perform image captioning on the image
        caption = get_caption([image_array], num_return_sequences)

        # Display the image and its caption
        st.image(image, use_column_width=True, caption="Original Image")
        for i, c in enumerate(caption):
            st.write(f"Caption {i + 1}: {c}")