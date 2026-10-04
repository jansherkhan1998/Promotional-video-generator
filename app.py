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
        groq_key = st.secrets.get("GROQ_API_KEY")
        hf_token = st.secrets.get("HF_TOKEN")

        if not hf_token:
            st.error("HF_TOKEN is missing in Streamlit Secrets!")
        else:
            # Save uploaded image to temp file for Gradio client
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            # Step 1: Refine Prompt via Groq Cloud API (Llama 3)
            enhanced_prompt = user_prompt
            if groq_key:
                with st.spinner("Step 1/2: Enhancing description with Groq (Llama 3)..."):
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
                                    "content": "You are a professional film director. Rewrite the user description into a single concise cinematic prompt for an image-to-video AI model."
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
                            st.info(f"Groq API status {res.status_code}. Using raw description.")
                    except Exception as e:
                        st.info(f"Groq call skipped ({e}). Using raw description.")
            else:
                st.info("GROQ_API_KEY not found in secrets, using raw description.")

           # Step 2: Render Video on Hugging Face Wan 2.1 Space
            with st.spinner("Step 2/2: Rendering video on Hugging Face (Takes ~60s)..."):
                try:
                    # Connect to the Hugging Face Space
                    hf_client = Client("multimodalart/wan2-1-fast", token=hf_token)
                    
                    # Method A: Try auto-detecting default function execution
                    result = hf_client.predict(
                        handle_file(tmp_path), # Image file
                        enhanced_prompt        # Refined text prompt
                    )
                    
                    st.success("Rendering Complete!")
                    st.video(result)

                except Exception as e:
                    # Method B: Fallback to official Wan-AI zero-GPU space
                    try:
                        st.info("Swapping to Wan-AI official GPU queue...")
                        wan_client = Client("Wan-AI/Wan2.1", token=hf_token)
                        result = wan_client.predict(
                            handle_file(tmp_path),
                            enhanced_prompt
                        )
                        st.success("Rendering Complete!")
                        st.video(result)
                    except Exception as err:
                        st.error(f"Generation error: {err}. Hugging Face GPUs are experiencing high traffic. Please wait 30 seconds and click Generate again.")
                    st.success("Rendering Complete!")
                    st.video(result)
                except Exception as e:
                    st.error(f"Generation error: {e}. Hugging Face GPUs might be experiencing heavy queue traffic. Please retry in 1 minute.")
