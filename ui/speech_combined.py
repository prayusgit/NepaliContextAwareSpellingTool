import streamlit as st
import speech_recognition as sr
import paragraph  # Import paragraph.py
import sentence  # Import sentence.py


# Speech-to-text function for Nepali
def speech_to_text():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        st.write("Please speak in Nepali...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        # Recognizing speech using Google Web Speech API
        text = recognizer.recognize_google(audio, language='ne-NP')  # Nepali language code
        st.write("You said: ", text)
        return text
    except sr.UnknownValueError:
        st.write("Sorry, I could not understand your speech.")
        return None
    except sr.RequestError:
        st.write("Could not request results from Google Web Speech API.")
        return None


def main():

    mode = st.sidebar.selectbox("Select Mode", ("Sentence", "Paragraph", "Speech-to-Text"))

    if mode == "Sentence":
        st.write("### Sentence Mode")
        sentence.main()  # Call the sentence mode function

    elif mode == "Paragraph":
        st.write("### Paragraph Mode")
        paragraph.main()  # Call the paragraph mode function

    elif mode == "Speech-to-Text":
        st.write("### Speech-to-Text Mode")
        speech_text = speech_to_text()

        if speech_text:
            st.write("### Corrected Sentence:")
            # Process the speech text as you would with the sentence mode
            sentence.main()  # Alternatively, you can pass speech_text to sentence.main if needed.


if __name__ == "__main__":
    main()
