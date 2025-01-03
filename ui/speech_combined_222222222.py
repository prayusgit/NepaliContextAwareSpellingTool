import streamlit as st
import speech_recognition as sr
import paragraph  # Import paragraph.py
import sentence  # Import sentence.py
import ui2


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
        ui2.speech_to_text()


if __name__ == "__main__":
    main()
