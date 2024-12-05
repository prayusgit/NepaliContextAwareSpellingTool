import streamlit as st
import pyttsx3
import speech_recognition as sr

# Initialize recognizer and text-to-speech engine
recognizer = sr.Recognizer()
engine = pyttsx3.init()


# Function to recognize speech
def recognize_audio():
    with sr.Microphone() as source:
        st.write("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio, language='ne-IN')
            return text
        except sr.UnknownValueError:
            return "Sorry, I could not understand the audio."
        except sr.RequestError:
            return "Sorry, the service is unavailable."


# Streamlit UI
st.title("Nepali Spelling Correction with Audio Input")

# Display input field
original_sentence = st.text_input("Enter a Nepali sentence:", "")

# Button to trigger audio input
if st.button("Listen to Audio"):
    # Recognize speech and update the text field
    recognized_text = recognize_audio()
    if recognized_text:
        st.text_input("Recognized Text", recognized_text)  # This copies the recognized text directly to the input box

    # Process the recognized text (you can integrate the spelling correction here)
    corrected_text = recognized_text  # Use your spelling correction model here if needed

    # Optionally, you can also use text-to-speech to say the corrected text
    engine.say(corrected_text)
    engine.runAndWait()
