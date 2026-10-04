import streamlit as st
from google import genai
from gradio_client import Client, handle_file
import tempfile

st.set_page_config(page_title="Promotional Video Generator", layout="centered")

st.title("🎬 Promotional Video Generator")
st.write("Generate realistic AI videos with $0 cost using open-source models.")

# 1. Inputs
uploaded_file = st.file_uploader("Upload starting image", type=["png", "jpg", "jpeg"])
user_prompt = st.text_area("Describe the motion & scene", "A dramatic cinematic slow-motion pan across cosmetic bottles on a wet marble surface.")

if st.button("Generate Video (100% Free)", type="primary"):
    if not uploaded_file:
        st.error("Please upload an image first!")
    else:
        # Load API keys from Streamlit Secrets
        gemini_key = st.secrets.get("GEMINI_API_KEY")
        hf_token = st.secrets.get("HF_TOKEN")

        if not hf_token:
            st.error("HF_TOKEN is missing in Streamlit Secrets!")
        else:
            # Save uploaded image to temp file for Gradio client
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            # Step 1: Enhance prompt using Gemini API (if available)
            enhanced_prompt = user_prompt
            if gemini_key:
                with st.spinner("Step 1/2: Enhancing description with Gemini..."):
                    try:
                        ai_client = genai.Client(api_key=gemini_key)
                        prompt_response = ai_client.models.generate_content(
                            model="gemini-2.0-flash",
                            contents=f"Rewrite this user description into a detailed cinematic image-to-video prompt: {user_prompt}"
                        )
                        enhanced_prompt = prompt_response.text
                        st.success(f"**Enhanced Prompt:** {enhanced_prompt}")
                    except Exception as e:
                        st.warning("Could not enhance prompt, using raw description.")
            else:
                st.warning("GEMINI_API_KEY not found in secrets, using raw description.")

            # Step 2: Generate video via active Hugging Face Space
            with st.spinner("Step 2/2: Rendering video on Hugging Face (Takes ~60s)..."):
                try:
                    # Pointing to an active Gradio space endpoint
                    hf_client = Client("Wan-AI/Wan2.1", token=hf_token)
                    
                    result = hf_client.predict(
                        image=handle_file(tmp_path),
                        prompt=enhanced_prompt,
                        api_name="/generate_video"
                    )
                    
                    st.success("Rendering Complete!")
                    st.video(result)
                except Exception as e:
                    st.error(f"Generation error: {e}. Hugging Face GPUs might be busy or in queue. Please retry in a moment.")
