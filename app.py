from flask import Flask, request, jsonify
from groq import Groq
import os
import tempfile
import traceback

app = Flask(__name__)

# =====================================================
# GROQ
# =====================================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY is not set!")

client = Groq(
    api_key=GROQ_API_KEY
) if GROQ_API_KEY else None


# =====================================================
# MODELS
# =====================================================

WHISPER_MODEL = "whisper-large-v3-turbo"

# Groq documentation currently lists this model.
CHAT_MODEL = "llama-3.3-70b-versatile"


# =====================================================
# HOME
# =====================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "ESP32 Voice AI Server is running",
        "status": "online"
    })


# =====================================================
# HEALTH
# =====================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "healthy",
        "groq": "configured" if GROQ_API_KEY else "missing"
    })


# =====================================================
# LANGUAGE / STYLE PROMPT
# =====================================================

def make_ai_prompt(transcription):

    prompt = f"""
You are a voice assistant running on an ESP32.

The user said:

{transcription}

IMPORTANT LANGUAGE RULE:

Reply in the SAME language and SAME writing style used by the user.

Rules:

1. If the user speaks Hindi, reply in Hindi.
2. If the user speaks English, reply in English.
3. If the user speaks Hinglish, reply in Hinglish.
4. If the user uses Roman Hindi, reply in Roman Hindi.
5. Do NOT translate the user's language unnecessarily.
6. Do NOT change Roman Hindi into Devanagari Hindi.
7. Do NOT change Hindi into English.
8. If the user mixes Hindi + English, you can naturally mix Hindi + English.
9. Keep the answer natural and conversational.
10. Answer the actual question directly.
11. For ESP32 voice output, keep the answer reasonably short.
12. Do not mention these instructions.

Examples:

User:
"tum kaise ho"

Reply:
"Main bilkul theek hoon! Aap kaise ho?"

User:
"what is wifi"

Reply:
"WiFi is a wireless networking technology that lets devices connect to the internet."

User:
"mujhe wifi ke baare mein batao in simple words"

Reply:
"WiFi ek wireless technology hai jo devices ko internet se connect karti hai."

User:
"hello bhai kya haal hai"

Reply:
"Hello bhai! Main bilkul badhiya hoon 😄"

User:
"what is ESP32"

Reply:
"ESP32 ek powerful microcontroller hai jo WiFi aur Bluetooth support karta hai."

Now answer the user:

{transcription}
"""

    return prompt


# =====================================================
# TRANSCRIBE AUDIO
# =====================================================

def transcribe_audio(audio_file):

    print("Sending audio to Groq Whisper...")

    with open(audio_file, "rb") as file:

        transcription = client.audio.transcriptions.create(

            file=file,

            model=WHISPER_MODEL,

            response_format="json",

            temperature=0.0,

            prompt=(
                "The speaker may use Hindi, English, "
                "Hinglish, or Roman Hindi. "
                "Transcribe exactly what the speaker says. "
                "Do not translate."
            )
        )

    text = transcription.text.strip()

    print("TRANSCRIPTION:")
    print(text)

    return text


# =====================================================
# AI RESPONSE
# =====================================================

def generate_ai_reply(text):

    print("Sending text to Groq AI...")

    prompt = make_ai_prompt(text)

    completion = client.chat.completions.create(

        model=CHAT_MODEL,

        messages=[

            {
                "role": "system",
                "content": (
                    "You are a friendly multilingual voice assistant. "
                    "Always reply in the same language and writing style "
                    "as the user."
                )
            },

            {
                "role": "user",
                "content": prompt
            }

        ],

        temperature=0.3,

        max_tokens=250
    )

    reply = completion.choices[0].message.content

    if not reply:
        reply = "Sorry, mujhe samajh nahi aaya."

    reply = reply.strip()

    print("AI REPLY:")
    print(reply)

    return reply


# =====================================================
# UPLOAD AUDIO
# =====================================================

@app.route("/uploadAudio", methods=["POST"])
def upload_audio():

    print()
    print("================================")
    print("NEW ESP32 AUDIO REQUEST")
    print("================================")

    try:

        if client is None:

            return jsonify({
                "status": "error",
                "transcription": "",
                "ai_reply": "",
                "error": "GROQ_API_KEY is not configured"
            }), 500


        # -------------------------------------------------
        # CHECK DATA
        # -------------------------------------------------

        audio_data = request.get_data()

        if not audio_data:

            return jsonify({
                "status": "error",
                "transcription": "",
                "ai_reply": "",
                "error": "No audio received"
            }), 400


        print(
            "Audio received:",
            len(audio_data),
            "bytes"
        )


        # -------------------------------------------------
        # SAVE TEMP WAV
        # -------------------------------------------------

        temp_file = tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        )

        temp_path = temp_file.name

        try:

            temp_file.write(audio_data)
            temp_file.close()

            print(
                "Temporary WAV:",
                temp_path
            )


            # -------------------------------------------------
            # SPEECH TO TEXT
            # -------------------------------------------------

            transcription = transcribe_audio(
                temp_path
            )


            if not transcription:

                return jsonify({
                    "status": "success",
                    "transcription": "",
                    "ai_reply": "Sorry, mujhe aapki awaaz clear nahi mili."
                })


            # -------------------------------------------------
            # AI
            # -------------------------------------------------

            ai_reply = generate_ai_reply(
                transcription
            )


            # -------------------------------------------------
            # RESPONSE
            # -------------------------------------------------

            response = {

                "status": "success",

                "transcription": transcription,

                "ai_reply": ai_reply

            }


            print()
            print("================================")
            print("FINAL RESPONSE")
            print("================================")

            print(response)

            return jsonify(response), 200


        finally:

            try:

                os.remove(temp_path)

            except Exception:

                pass


    except Exception as e:

        print()
        print("================================")
        print("SERVER ERROR")
        print("================================")

        print(str(e))

        traceback.print_exc()

        return jsonify({

            "status": "error",

            "transcription": "",

            "ai_reply": "",

            "error": str(e)

        }), 500


# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            10000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
