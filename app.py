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
        # Load API keys from Streamlit Secrets
        groq_key = st.secrets.get("GROQ_API_KEY")
        hf_token = st.secrets.get("HF_TOKEN")

        if not hf_token:
            st.error("HF_TOKEN is missing in Streamlit Secrets!")
        else:
            # Save uploaded image to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            # Step 1: Refine Prompt via Groq Cloud API
            enhanced_prompt = user_prompt
            if groq_key:
                with st.spinner("Step 1/2: Enhancing description with Groq..."):
                    try:
                        headers = {
                            "Authorization": f"Bearer {groq_key}",
                            "Content-Type": "application/json"
                        }
                        payload = {
                            "model": "openai/gpt-oss-120b",
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "You are a film director. Rewrite user text into a detailed cinematic prompt for an image-to-video AI."
                                },
                                {
                                    "role": "user",
                                    "content": user_prompt
                                }
                            ]
                        }
                        res = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=10)
                        if res.status_code == 200:
                            enhanced_prompt = res.json()["choices"][0]["message"]["content"]
                            st.success(f"**Enhanced Prompt:** {enhanced_prompt}")
                        else:
                            st.info("Using raw description for video generation.")
                    except Exception:
                        st.info("Using raw description for video generation.")

            # Step 2: Render Video on Hugging Face ZeroGPU Space
            with st.spinner("Step 2/2: Rendering video on Hugging Face (Takes ~60s)..."):
                video_url_or_path = None
                
                # Attempt Primary Space
                try:
                    hf_client = Client("multimodalart/wan2-1-fast", token=hf_token)
                    video_url_or_path = hf_client.predict(
                        prompt=enhanced_prompt,
                        image=handle_file(tmp_path),
                        api_name="/predict"
                    )
                except Exception as e1:
                    # Attempt Fallback Space if primary endpoint or queue fails
                    try:
                        st.info("Primary GPU queue busy, switching to backup space...")
                        fallback_client = Client("Wan-AI/Wan2.1", token=hf_token)
                        video_url_or_path = fallback_client.predict(
                            prompt=enhanced_prompt,
                            image=handle_file(tmp_path),
                            api_name="/generate"
                        )
                    except Exception as e2:
                        st.error(f"Generation error: {e2}. Hugging Face public GPUs are experiencing heavy traffic. Please wait 30 seconds and click Generate again.")

                # Render video only if output was successfully produced
                if video_url_or_path:
                    st.success("Rendering Complete!")
                    st.video(video_url_or_path)
