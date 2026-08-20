from flask import Flask, request, jsonify
from groq import Groq
import speech_recognition as sr
import os
import tempfile
import traceback

app = Flask(__name__)

# =====================================================
# GROQ
# =====================================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("ERROR: GROQ_API_KEY missing")

groq_client = Groq(
    api_key=GROQ_API_KEY
) if GROQ_API_KEY else None


# =====================================================
# MODELS
# =====================================================

WHISPER_MODEL = "whisper-large-v3-turbo"
CHAT_MODEL = "llama-3.3-70b-versatile"


# =====================================================
# HOME
# =====================================================

@app.route("/")
def home():
    return jsonify({
        "message": "ESP32 Voice AI Server is running",
        "status": "online"
    })


# =====================================================
# HEALTH
# =====================================================

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "groq": bool(GROQ_API_KEY)
    })


# =====================================================
# SPEECH RECOGNITION
# =====================================================

def recognize_speech(wav_path):

    print("Starting speech recognition...")

    recognizer = sr.Recognizer()

    try:

        with sr.AudioFile(wav_path) as source:

            # WAV ko properly read karega
            audio = recognizer.record(source)

        print("Audio loaded successfully")

    except Exception as e:

        print("Audio read error:", e)
        raise


    # -------------------------------------------------
    # First: Groq Whisper
    # -------------------------------------------------

    try:

        print("Sending WAV to Groq Whisper...")

        with open(wav_path, "rb") as audio_file:

            result = groq_client.audio.transcriptions.create(
                file=audio_file,
                model=WHISPER_MODEL,
                response_format="json",
                temperature=0.0,
                prompt=(
                    "The speaker can speak Hindi, English, "
                    "Hinglish or Roman Hindi. "
                    "Transcribe exactly what the speaker says. "
                    "Do not translate."
                )
            )

        text = result.text.strip()

        print("Groq transcription:", text)

        return text

    except Exception as e:

        print("Groq STT error:", e)

        # -------------------------------------------------
        # Fallback: Google Speech Recognition
        # -------------------------------------------------

        try:

            print("Trying SpeechRecognition fallback...")

            text = recognizer.recognize_google(
                audio
            )

            print(
                "SpeechRecognition result:",
                text
            )

            return text.strip()

        except sr.UnknownValueError:

            print("Speech could not be understood")

            return ""

        except sr.RequestError as e:

            print(
                "SpeechRecognition network error:",
                e
            )

            return ""


# =====================================================
# AI RESPONSE
# =====================================================

def generate_reply(text):

    prompt = f"""
You are a friendly ESP32 voice assistant.

User said:

{text}

LANGUAGE RULES:

- Hindi input -> Hindi reply.
- English input -> English reply.
- Hinglish input -> Hinglish reply.
- Roman Hindi input -> Roman Hindi reply.
- Hindi + English mixed input -> naturally mix Hindi + English.
- Never unnecessarily translate.
- If user writes Hindi using English letters,
  reply using English letters too.
- Keep the response short and natural.
- Answer what the user actually asked.

Examples:

User:
tum kaise ho

Assistant:
Main bilkul theek hoon bhai! Aap kaise ho?

User:
what is esp32

Assistant:
ESP32 is a microcontroller with built-in WiFi and Bluetooth.

User:
bhai ESP32 kya hai

Assistant:
ESP32 ek powerful microcontroller hai jisme WiFi aur Bluetooth built-in hota hai.

User:
hello bhai what are you doing

Assistant:
Hello bhai! Main aapki help karne ke liye ready hoon.

Now reply to:

{text}
"""

    response = groq_client.chat.completions.create(

        model=CHAT_MODEL,

        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful multilingual voice assistant. "
                    "Always match the user's language and writing style."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.3,
        max_tokens=200
    )

    reply = response.choices[0].message.content

    if not reply:
        reply = "Sorry, mujhe samajh nahi aaya."

    return reply.strip()


# =====================================================
# UPLOAD AUDIO
# =====================================================

@app.route("/uploadAudio", methods=["POST"])
def upload_audio():

    print()
    print("================================")
    print("NEW ESP32 RECORDING")
    print("================================")

    wav_data = request.get_data()

    if not wav_data:

        return jsonify({
            "status": "error",
            "transcription": "",
            "ai_reply": "",
            "error": "No audio received"
        }), 400


    print(
        "Received WAV:",
        len(wav_data),
        "bytes"
    )


    temp_path = None

    try:

        # -------------------------------------------------
        # SAVE WAV
        # -------------------------------------------------

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as temp:

            temp.write(wav_data)

            temp_path = temp.name


        print(
            "WAV saved:",
            temp_path
        )


        # -------------------------------------------------
        # CHECK WAV USING SpeechRecognition
        # -------------------------------------------------

        transcription = recognize_speech(
            temp_path
        )


        if not transcription:

            return jsonify({
                "status": "success",
                "transcription": "",
                "ai_reply": (
                    "Mujhe aapki awaaz clear nahi mili, "
                    "please dobara bolo."
                )
            })


        # -------------------------------------------------
        # GROQ AI
        # -------------------------------------------------

        print("Generating AI reply...")

        ai_reply = generate_reply(
            transcription
        )


        # -------------------------------------------------
        # RESPONSE
        # -------------------------------------------------

        result = {

            "status": "success",

            "transcription": transcription,

            "ai_reply": ai_reply
        }


        print()
        print("================================")
        print("RESULT")
        print("================================")

        print("Text:", transcription)
        print("AI:", ai_reply)


        return jsonify(result)


    except Exception as e:

        print()
        print("================================")
        print("ERROR")
        print("================================")

        print(str(e))

        traceback.print_exc()

        return jsonify({

            "status": "error",

            "transcription": "",

            "ai_reply": "",

            "error": str(e)

        }), 500


    finally:

        if temp_path:

            try:
                os.remove(temp_path)
            except:
                pass


# =====================================================
# START
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
