import streamlit as st
import requests
from gradio_client import Client, handle_file
import tempfile

st.set_page_config(page_title="Promotional Video Generator", layout="centered")

st.title("🎬 Promotional Video Generator")
st.write("Generate realistic AI videos with $0 cost using open-source models.")

# 1. Inputs
uploaded_file = st.file_uploader("Upload starting image", type=["png", "jpg", "jpeg"])
user_prompt = st.text_area(
    "Describe the motion & scene", 
    "A dramatic cinematic slow-motion pan across cosmetic tubes sitting on a wet marble surface, golden hour lighting."
)

if st.button("Generate Video (100% Free)", type="primary"):
    if not uploaded_file:
        st.error("Please upload an image first!")
    else:
        # Load keys from Streamlit Secrets
        gemini_key = st.secrets.get("GEMINI_API_KEY")
        hf_token = st.secrets.get("HF_TOKEN")

        if not hf_token:
            st.error("HF_TOKEN is missing in Streamlit Secrets!")
        else:
            # Save uploaded image to temp file for Gradio client
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            # Step 1: Refine Prompt via Gemini API REST call (bypasses SDK version conflicts)
            enhanced_prompt = user_prompt
            if gemini_key:
                with st.spinner("Step 1/2: Enhancing description with Gemini..."):
                    try:
                        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_key}"
                        payload = {
                            "contents": [{
                                "parts": [{
                                    "text": f"Rewrite this user description into a detailed cinematic image-to-video prompt: {user_prompt}"
                                }]
                            }]
                        }
                        res = requests.post(url, json=payload, timeout=10)
                        if res.status_code == 200:
                            data = res.json()
                            enhanced_prompt = data['candidates'][0]['content']['parts'][0]['text']
                            st.success(f"**Enhanced Prompt:** {enhanced_prompt}")
                        else:
                            st.info("Using raw description for generation.")
                    except Exception:
                        st.info("Using raw description for generation.")

            # Step 2: Render Video on Hugging Face (Auto-discovers endpoints)
            with st.spinner("Step 2/2: Rendering video on Hugging Face (Takes ~60s)..."):
                try:
                    # Connects to active Hugging Face ZeroGPU space
                    hf_client = Client("Wan-AI/Wan2.1", token=hf_token)
                    
                    # Passing inputs without hardcoding restrictive api_name strings
                    result = hf_client.predict(
                        handle_file(tmp_path),
                        enhanced_prompt
                    )
                    
                    st.success("Rendering Complete!")
                    st.video(result)
                except Exception as e:
                    # Alternative Space Fallback if the primary space queue is full
                    try:
                        st.warning("Primary space busy, routing to fast fallback GPU space...")
                        fallback_client = Client("multimodalart/wan2-1-fast", token=hf_token)
                        result = fallback_client.predict(
                            handle_file(tmp_path),
                            enhanced_prompt
                        )
                        st.success("Rendering Complete!")
                        st.video(result)
                    except Exception as fallback_err:
                        st.error(f"Generation error: {fallback_err}. Hugging Face GPUs might be experiencing heavy queue traffic. Please retry in 1 minute.")
