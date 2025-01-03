import streamlit as st
import pyttsx3
import speech_recognition as sr
from sentence import *

# Initialize recognizer and text-to-speech engine
def speech_to_text():
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

        tokenizer, model, device = load_model()
        confusion_set = load_confusion_set()

        # User input
        if st.button("Check Spelling"):
            if recognized_text.strip():
                corrected_sentence, word_probabilities, corrections = spell_check_with_probabilities(
                    recognized_text, confusion_set, tokenizer, model, device
                )
                st.write("### Possibilities of error")
                highlighted_possible_error = highlight_possible_error(recognized_text, confusion_set.keys())
                st.markdown(highlighted_possible_error, unsafe_allow_html=True)

                st.write("### Corrected Sentence:")
                highlighted = highlight_corrections(recognized_text, corrected_sentence)
                st.markdown(highlighted, unsafe_allow_html=True)

                if word_probabilities:
                    st.write("### Word Probabilities:")
                    for word, probs in word_probabilities.items():
                        st.write(f"**{word}:**")
                        # Sort probabilities in descending order
                        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                        for candidate, prob in sorted_probs:
                            st.write(f"- {candidate}: {prob}")
            else:
                st.warning("Please enter a sentence before clicking the button.")
        # Optionally, you can also use text-to-speech to say the corrected text
            engine.say(corrected_sentence)
            engine.runAndWait()

