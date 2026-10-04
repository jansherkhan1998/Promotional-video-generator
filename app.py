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

          # Step 2: Render Video on Hugging Face (LTX-Video Model)
            with st.spinner("Step 2/2: Rendering video with LTX-Video (~30s)..."):
                video_url_or_path = None
                try:
                    # Connect to the official LTX-Video Space
                    hf_client = Client("Lightricks/LTX-Video", token=hf_token)
                    
                    # Generate video using LTX-Video endpoint
                    result = hf_client.predict(
                        prompt=enhanced_prompt,
                        image=handle_file(tmp_path),
                        negative_prompt="worst quality, blurry, distorted, low resolution",
                        frame_rate=25,
                        guidance_scale=3.0,
                        num_inference_steps=30,
                        seed=42,
                        api_name="/generate"
                    )
                    
                    # Extract result path safely
                    if isinstance(result, (tuple, list)):
                        video_url_or_path = result[0]
                    else:
                        video_url_or_path = result

                except Exception as e:
                    st.error(f"Generation error: {e}. Hugging Face GPUs might be queued. Please wait 15 seconds and try again.")

                # Render video player if generation succeeded
                if video_url_or_path:
                    st.success("Rendering Complete!")
                    st.video(video_url_or_path)
