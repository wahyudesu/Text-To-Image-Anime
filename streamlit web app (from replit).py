import streamlit as st
import torch
from diffusers import DiffusionPipeline

# Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    pipe = DiffusionPipeline.from_pretrained(
        "cagliostrolab/animagine-xl-3.1",
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to("cuda")
    pipe.load_lora_weights(".", weight_name="deltamon_beta.safetensors")
    return pipe

# Function to generate the image
def generate_image(prompt, negative_prompt):
    pipe = load_model()
    image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        width=832,
        height=1216,
        guidance_scale=7,
        num_inference_steps=28,
    ).images[0]
    return image

# Streamlit app
st.title("Image Generator")

# Input fields for prompt and negative prompt
prompt = st.text_input("Enter your prompt:")
negative_prompt = st.text_input("Enter your negative prompt:")

# Button to generate the image
if st.button("Generate Image"):
    if prompt and negative_prompt:
        image = generate_image(prompt, negative_prompt)
        st.image(image, caption="Generated Image", use_column_width=True)
    else:
        st.error("Please enter both prompt and negative prompt.")