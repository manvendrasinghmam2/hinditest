from flask import Flask, request, jsonify
from groq import Groq
import speech_recognition as sr
import os
import tempfile
import traceback
import wave

app = Flask(__name__)

# =====================================================
# GROQ
# =====================================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY is missing")

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

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "status": "online",
        "message": "ESP32 Voice AI Server",
        "upload_endpoint": "/uploadAudio",
        "health_endpoint": "/health"
    })


# =====================================================
# HEALTH
# =====================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "healthy",
        "groq_configured": groq_client is not None,
        "whisper_model": WHISPER_MODEL,
        "chat_model": CHAT_MODEL
    })


# =====================================================
# TEST UPLOAD ROUTE
# =====================================================

@app.route("/uploadAudio", methods=["POST"])
def upload_audio():

    print()
    print("========================================")
    print("NEW ESP32 AUDIO REQUEST")
    print("========================================")

    try:

        # -------------------------------------------------
        # READ RAW BODY
        # -------------------------------------------------

        wav_data = request.get_data()

        print(
            "Received bytes:",
            len(wav_data)
        )

        if not wav_data:

            return jsonify({
                "status": "error",
                "transcription": "",
                "ai_reply": "",
                "error": "No audio received"
            }), 400


        # -------------------------------------------------
        # BASIC WAV CHECK
        # -------------------------------------------------

        if len(wav_data) < 44:

            return jsonify({
                "status": "error",
                "transcription": "",
                "ai_reply": "",
                "error": "Invalid WAV: file too small"
            }), 400


        if wav_data[0:4] != b"RIFF":

            return jsonify({
                "status": "error",
                "transcription": "",
                "ai_reply": "",
                "error": "Invalid WAV: RIFF header missing"
            }), 400


        if wav_data[8:12] != b"WAVE":

            return jsonify({
                "status": "error",
                "transcription": "",
                "ai_reply": "",
                "error": "Invalid WAV: WAVE header missing"
            }), 400


        print("WAV header: OK")


        # -------------------------------------------------
        # SAVE TEMP WAV
        # -------------------------------------------------

        temp_path = None

        try:

            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False
            ) as temp:

                temp.write(wav_data)

                temp_path = temp.name


            print(
                "Temporary WAV:",
                temp_path
            )


            # -------------------------------------------------
            # CHECK WAV PARAMETERS
            # -------------------------------------------------

            try:

                with wave.open(
                    temp_path,
                    "rb"
                ) as wav:

                    channels = wav.getnchannels()
                    sample_width = wav.getsampwidth()
                    sample_rate = wav.getframerate()
                    frames = wav.getnframes()

                    duration = (
                        frames / sample_rate
                        if sample_rate > 0
                        else 0
                    )

                    print()
                    print("WAV INFO")
                    print("--------------------")
                    print("Channels:", channels)
                    print("Sample width:", sample_width)
                    print("Sample rate:", sample_rate)
                    print("Frames:", frames)
                    print("Duration:", duration)
                    print("--------------------")


            except Exception as e:

                print(
                    "WAV inspection warning:",
                    e
                )


            # -------------------------------------------------
            # GROQ CHECK
            # -------------------------------------------------

            if groq_client is None:

                return jsonify({
                    "status": "error",
                    "transcription": "",
                    "ai_reply": "",
                    "error": "GROQ_API_KEY missing on Render"
                }), 500


            # -------------------------------------------------
            # SPEECH TO TEXT
            # -------------------------------------------------

            print()
            print("Sending audio to Groq Whisper...")

            try:

                with open(
                    temp_path,
                    "rb"
                ) as audio_file:

                    transcription_result = (
                        groq_client.audio.transcriptions.create(

                            file=audio_file,

                            model=WHISPER_MODEL,

                            response_format="json",

                            temperature=0.0,

                            prompt=(
                                "The speaker may use Hindi, "
                                "English, Hinglish, or Roman Hindi. "
                                "Transcribe exactly what is spoken. "
                                "Do not translate Hindi into English. "
                                "Do not translate English into Hindi. "
                                "Keep Roman Hindi in Roman letters."
                            )
                        )
                    )


                transcription = (
                    transcription_result.text or ""
                ).strip()


                print()
                print("TRANSCRIPTION:")
                print(transcription)


            except Exception as e:

                print()
                print("WHISPER ERROR:")
                print(str(e))

                traceback.print_exc()

                return jsonify({

                    "status": "error",

                    "transcription": "",

                    "ai_reply": "",

                    "error": (
                        "Speech recognition failed: "
                        + str(e)
                    )

                }), 500


            # -------------------------------------------------
            # EMPTY SPEECH
            # -------------------------------------------------

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
            # AI RESPONSE
            # -------------------------------------------------

            print()
            print("Generating AI response...")

            try:

                ai_reply = generate_reply(
                    transcription
                )

            except Exception as e:

                print()
                print("AI ERROR:")
                print(str(e))

                traceback.print_exc()

                return jsonify({

                    "status": "error",

                    "transcription": transcription,

                    "ai_reply": "",

                    "error": (
                        "AI response failed: "
                        + str(e)
                    )

                }), 500


            # -------------------------------------------------
            # FINAL RESPONSE
            # -------------------------------------------------

            print()
            print("========================================")
            print("FINAL RESULT")
            print("========================================")

            print(
                "Text:",
                transcription
            )

            print(
                "AI:",
                ai_reply
            )

            print(
                "========================================"
            )


            return jsonify({

                "status": "success",

                "transcription": transcription,

                "ai_reply": ai_reply

            })


        finally:

            # -------------------------------------------------
            # DELETE TEMP FILE
            # -------------------------------------------------

            if temp_path:

                try:

                    os.remove(
                        temp_path
                    )

                except Exception:

                    pass


    except Exception as e:

        print()
        print("========================================")
        print("SERVER ERROR")
        print("========================================")

        print(
            str(e)
        )

        traceback.print_exc()


        return jsonify({

            "status": "error",

            "transcription": "",

            "ai_reply": "",

            "error": str(e)

        }), 500


# =====================================================
# AI
# =====================================================

def generate_reply(user_text):

    system_prompt = """
You are a friendly voice assistant running on an ESP32.

The user can speak:

1. English
2. Hindi
3. Hinglish
4. Roman Hindi
5. Hindi + English mixed
6. English + Hindi mixed

IMPORTANT LANGUAGE RULES:

- Reply in the SAME language/style as the user.
- English input -> English reply.
- Hindi spoken/written in Devanagari -> Hindi Devanagari reply.
- Roman Hindi -> Roman Hindi.
- Hinglish -> Hinglish.
- Hindi + English mixed -> naturally mix Hindi and English.
- Do NOT unnecessarily translate.
- Do NOT change Roman Hindi into Devanagari.
- Do NOT change English into Hindi unless the user does so.
- Keep replies short because this is an ESP32 voice assistant.
- Normally answer in 1 to 3 short sentences.
- Be natural and conversational.
- Understand spelling mistakes caused by speech recognition.
- If the transcription is slightly wrong, infer the most likely meaning.

Examples:

User:
hello how are you

Assistant:
I'm good! How can I help you?

User:
tum kaise ho

Assistant:
Main bilkul theek hoon! Aap kaise ho?

User:
bhai tum kaise ho

Assistant:
Main bilkul theek hoon bhai! Batao kya help chahiye?

User:
what is esp32

Assistant:
ESP32 ek powerful microcontroller hai with built-in WiFi and Bluetooth.

User:
ESP32 kya hai

Assistant:
ESP32 ek powerful microcontroller hai jisme WiFi aur Bluetooth built-in hota hai.

User:
mujhe weather batao

Assistant:
Bilkul! Aap kis city ka weather jaana chahte ho?

User:
what is wifi

Assistant:
WiFi ek wireless technology hai jo devices ko internet ya local network se connect karti hai.

Now respond naturally to the user's exact message.
"""


    response = groq_client.chat.completions.create(

        model=CHAT_MODEL,

        messages=[

            {
                "role": "system",
                "content": system_prompt
            },

            {
                "role": "user",
                "content": user_text
            }

        ],

        temperature=0.2,

        max_tokens=150
    )


    reply = (
        response
        .choices[0]
        .message
        .content
    )


    if not reply:

        return (
            "Sorry, mujhe samajh nahi aaya."
        )


    return reply.strip()


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

    print()
    print("========================================")
    print("ESP32 VOICE AI SERVER")
    print("========================================")

    print(
        "Port:",
        port
    )

    print(
        "Whisper:",
        WHISPER_MODEL
    )

    print(
        "Chat:",
        CHAT_MODEL
    )

    print(
        "Groq:",
        "READY"
        if groq_client
        else "MISSING"
    )

    print(
        "========================================"
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
