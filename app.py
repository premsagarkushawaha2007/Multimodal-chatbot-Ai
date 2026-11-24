import streamlit as st
import os
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv
from gtts import gTTS
from io import BytesIO


# =========================================================
# LOAD API KEY
# =========================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Multimodal AI Chatbot",
    page_icon="🤖",
    layout="centered"
)


# =========================================================
# TITLE
# =========================================================

st.title("🤖 Multimodal AI Chatbot")

st.write(
    "Ask questions using Text, Images and Voice!"
)


# =========================================================
# CHECK API KEY
# =========================================================

if not API_KEY:

    st.error(
        "API Key not found. Please check your .env file."
    )

    st.stop()


# =========================================================
# GEMINI CONNECTION
# =========================================================

client = genai.Client(
    api_key=API_KEY
)


# Primary and fallback models
MODEL = "gemini-3.5-flash-lite"
FALLBACK_MODEL = "gemini-3.8-flash"


# =========================================================
# SESSION STATE
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []


if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# VOICE OUTPUT LANGUAGE
# =========================================================

language = st.selectbox(
    "Voice Output Language",
    ["Hindi", "English"]
)


# =========================================================
# VOICE OUTPUT FUNCTION
# =========================================================

def speak(text):

    try:

        if language == "Hindi":
            lang = "hi"

        else:
            lang = "en"

        audio = BytesIO()

        tts = gTTS(
            text=text,
            lang=lang
        )

        tts.write_to_fp(audio)

        audio.seek(0)

        st.audio(
            audio,
            format="audio/mp3"
        )

    except Exception as e:

        st.warning(
            "Voice output error: " + str(e)
        )


# =========================================================
# AI RESPONSE FUNCTION
# =========================================================

def ask_ai(new_parts):

    contents = []

    # -----------------------------------------------------
    # Add previous conversation
    # -----------------------------------------------------

    for item in st.session_state.history:

        contents.append(
            types.Content(
                role=item["role"],
                parts=[
                    types.Part.from_text(
                        text=item["text"]
                    )
                ]
            )
        )

    # -----------------------------------------------------
    # Add new user input
    # -----------------------------------------------------

    contents.append(
        types.Content(
            role="user",
            parts=new_parts
        )
    )

    # -----------------------------------------------------
    # Try primary model
    # -----------------------------------------------------

    try:

        response = client.models.generate_content(
            model=MODEL,
            contents=contents
        )

    # -----------------------------------------------------
    # If primary model gives 503
    # -----------------------------------------------------

    except Exception as e:

        error_message = str(e)

        if "503" in error_message:

            try:

                response = client.models.generate_content(
                    model=FALLBACK_MODEL,
                    contents=contents
                )

            except Exception as fallback_error:

                raise Exception(
                    "Both Gemini models are temporarily unavailable.\n\n"
                    "Primary model error: "
                    + error_message
                    + "\n\nFallback model error: "
                    + str(fallback_error)
                )

        else:

            raise e

    # -----------------------------------------------------
    # Get response text
    # -----------------------------------------------------

    answer = response.text

    if not answer:

        raise ValueError(
            "AI returned an empty response."
        )

    return answer


# =========================================================
# SHOW OLD CHAT MESSAGES
# =========================================================

for msg in st.session_state.messages:

    with st.chat_message(
        msg["role"]
    ):

        st.markdown(
            msg["text"]
        )


# =========================================================
# IMAGE UPLOAD
# =========================================================

st.subheader("📷 Image Upload")

image = st.file_uploader(
    "Upload an image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# Show uploaded image

if image:

    st.image(
        image,
        width=300
    )


# =========================================================
# TEXT CHAT
# =========================================================

prompt = st.chat_input(
    "Type your question..."
)


if prompt:

    # -----------------------------------------------------
    # Show user message
    # -----------------------------------------------------

    with st.chat_message("user"):

        st.markdown(
            prompt
        )

    # -----------------------------------------------------
    # Create text part
    # -----------------------------------------------------

    parts = [
        types.Part.from_text(
            text=prompt
        )
    ]

    # -----------------------------------------------------
    # Add image if uploaded
    # -----------------------------------------------------

    if image:

        parts.append(
            types.Part.from_bytes(
                data=image.getvalue(),
                mime_type=image.type
            )
        )

    # -----------------------------------------------------
    # Get AI answer
    # -----------------------------------------------------

    try:

        with st.spinner(
            "AI is thinking..."
        ):

            answer = ask_ai(
                parts
            )

        # -------------------------------------------------
        # Save conversation
        # -------------------------------------------------

        st.session_state.history.extend(
            [
                {
                    "role": "user",
                    "text": prompt
                },
                {
                    "role": "model",
                    "text": answer
                }
            ]
        )

        st.session_state.messages.extend(
            [
                {
                    "role": "user",
                    "text": prompt
                },
                {
                    "role": "assistant",
                    "text": answer
                }
            ]
        )

        # -------------------------------------------------
        # Show AI response
        # -------------------------------------------------

        with st.chat_message(
            "assistant"
        ):

            st.markdown(
                answer
            )

            # Voice output

            speak(answer)

    except Exception as e:

        st.error(
            "AI Error: " + str(e)
        )


# =========================================================
# VOICE INPUT
# =========================================================

st.divider()

st.subheader("🎤 Voice Input")

audio = st.audio_input(
    "Record your question"
)


if audio:

    st.audio(
        audio
    )

    if st.button(
        "Send Voice to AI"
    ):

        # -------------------------------------------------
        # Voice instruction
        # -------------------------------------------------

        voice_prompt = (
            "Listen to the attached audio. "
            "Understand the user's spoken question "
            "in Hindi or English and answer in the "
            "same language."
        )

        # -------------------------------------------------
        # Convert audio to Gemini part
        # -------------------------------------------------

        audio_part = types.Part.from_bytes(
            data=audio.getvalue(),
            mime_type="audio/wav"
        )

        parts = [
            types.Part.from_text(
                text=voice_prompt
            ),
            audio_part
        ]

        # -------------------------------------------------
        # Get AI answer
        # -------------------------------------------------

        try:

            with st.spinner(
                "Understanding your voice..."
            ):

                answer = ask_ai(
                    parts
                )

            # -------------------------------------------------
            # Save voice conversation
            # -------------------------------------------------

            st.session_state.history.extend(
                [
                    {
                        "role": "user",
                        "text":
                        "User asked a question using voice."
                    },
                    {
                        "role": "model",
                        "text": answer
                    }
                ]
            )

            st.session_state.messages.extend(
                [
                    {
                        "role": "user",
                        "text": "🎤 Voice message"
                    },
                    {
                        "role": "assistant",
                        "text": answer
                    }
                ]
            )

            # -------------------------------------------------
            # Show answer
            # -------------------------------------------------

            with st.chat_message(
                "assistant"
            ):

                st.markdown(
                    answer
                )

                speak(answer)

        except Exception as e:

            st.error(
                "Voice Error: " + str(e)
            )


# =========================================================
# CLEAR CHAT
# =========================================================

st.divider()

if st.button(
    "Clear Chat"
):

    st.session_state.history = []

    st.session_state.messages = []

    st.rerun()


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Created with Python + Streamlit + Gemini AI"
)