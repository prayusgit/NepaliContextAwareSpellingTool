import streamlit as st
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import json

# Load NepaliBERT
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("Shushant/nepaliBERT")
    model = AutoModelForMaskedLM.from_pretrained("Shushant/nepaliBERT")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device

# Load confusion set
@st.cache_data
def load_confusion_set():
    with open("confusion_set.json", "r", encoding="utf-8") as f:
        return json.load(f)

# Tokenize sentence
def tokenize_sentence(sentence, tokenizer, device):
    return tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(device)

# Get confusion candidates
def get_confusion_candidates(word, confusion_set):
    return confusion_set.get(word, [])

# Compute probabilities
def compute_probabilities(sentence, word, candidates, tokenizer, model, device):
    masked_sentence = sentence.replace(word, "[MASK]")
    inputs = tokenize_sentence(masked_sentence, tokenizer, device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        mask_index = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
        softmax = torch.nn.functional.softmax(logits[0, mask_index, :], dim=-1)
        probs = {candidate: softmax[0, tokenizer.convert_tokens_to_ids(candidate)].item() for candidate in candidates}
    return probs


# Spell-checker function with probabilities
def spell_check_with_probabilities(sentence, confusion_set, tokenizer, model, device):
    corrected_words = {}
    words = sentence.split()
    corrected_sentence = []
    word_probabilities = {}

    for word in words:
        if word in confusion_set:
            candidates = get_confusion_candidates(word, confusion_set)
            if candidates:
                probs = compute_probabilities(sentence, word, [word,*candidates], tokenizer, model, device)
                word_probabilities[word] = probs
                best_candidate = max(probs, key=probs.get)
                corrected_words[word] = best_candidate
                corrected_sentence.append(best_candidate)
            else:
                corrected_sentence.append(word)
        else:
            corrected_sentence.append(word)

    return " ".join(corrected_sentence), word_probabilities, corrected_words

def highlight_corrections(original_sentence, corrected_sentence):
    highlighted_text = corrected_sentence
    original_sentence_split = original_sentence.split()
    corrected_sentence_split = corrected_sentence.split()
    for original, corrected in zip(original_sentence_split, corrected_sentence_split):
        if original != corrected:
            highlighted_text = highlighted_text.replace(corrected, f"<span style='color: #9ABF80;'>{corrected}</span>")
    return highlighted_text



def highlight_possible_error(text, confusion_set):
    # Find the position of the word in the text
    split_sentence = text.split()

    for word in split_sentence:
        if word in confusion_set:
            text = text.replace(word, f"<span style='color: #FF6969;'>{word}</span>")

    return text


# Streamlit interface
def main():
    st.title("Nepali Spell Checker with Probabilities")
    st.write("Correct spelling mistakes in Nepali sentences.")

    # Load resources
    tokenizer, model, device = load_model()
    confusion_set = load_confusion_set()

    # User input
    sentence = st.text_area("Enter a Nepali sentence to check for errors:", "")

    if st.button("Check Spelling"):
        if sentence.strip():
            corrected_sentence, word_probabilities, corrections = spell_check_with_probabilities(
                sentence, confusion_set, tokenizer, model, device
            )
            st.write("### Possibilities of error")
            highlighted_possible_error = highlight_possible_error(sentence, confusion_set.keys())
            st.markdown(highlighted_possible_error, unsafe_allow_html=True)

            st.write("### Corrected Sentence:")
            highlighted = highlight_corrections(sentence, corrected_sentence)
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

    st.write("This tool uses NepaliBERT and a confusion set to suggest corrections.")

if __name__ == "__main__":
    main()
