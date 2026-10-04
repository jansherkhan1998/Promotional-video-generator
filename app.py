import streamlit as st
import openai
import replicate
import time

st.set_page_config(page_title="AI Video Generator", layout="centered")

st.title("🎬 AI Video Generator")
st.write("Upload a picture and enter a description to generate a realistic video.")

# 1. Inputs
uploaded_file = st.file_uploader("Upload starting image", type=["png", "jpg", "jpeg"])
user_prompt = st.text_area("Describe the scene & motion", "A dramatic cinematic shot with golden hour lighting, slow camera motion.")
aspect_ratio = st.selectbox("Aspect Ratio", ["16:9", "9:16", "1:1"])

# 2. Generation logic
if st.button("Generate Video", type="primary"):
    if not uploaded_file:
        st.error("Please upload an image first!")
    else:
        # Load API keys from Streamlit Secrets
        openai_key = st.secrets.get("OPENAI_API_KEY")
        replicate_key = st.secrets.get("REPLICATE_API_TOKEN")

        if not openai_key or not replicate_key:
            st.error("API Keys are missing in Streamlit Secrets!")
        else:
            with st.spinner("Step 1/2: Enhancing prompt with GPT-4o..."):
                client_ai = openai.OpenAI(api_key=openai_key)
                
                # Refine user text into a cinematic prompt
                system_instruction = "You are an expert director. Rewrite this user description into a detailed cinematic prompt for an AI video model."
                response = client_ai.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                enhanced_prompt = response.choices[0].message.content
                st.success(f"**Enhanced Prompt:** {enhanced_prompt}")

            with st.spinner("Step 2/2: Rendering video on Replicate (this takes 30–60s)..."):
                # Run the Replicate image-to-video model asynchronously
                try:
                    output = replicate.run(
                        "alibaba/wan-3",  # High quality video model hosted on Replicate
                        input={
                            "image": uploaded_file,
                            "prompt": enhanced_prompt,
                            "aspect_ratio": aspect_ratio
                        },
                        api_token=replicate_key
                    )
                    
                    # Display the generated video output
                    st.success("Video Render Complete!")
                    st.video(output)
                except Exception as e:
                    st.error(f"Error during video rendering: {e}")
