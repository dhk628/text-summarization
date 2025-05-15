from transformers import BartForConditionalGeneration, BartTokenizer
import torch
import streamlit as st


@st.cache_resource(show_spinner="Loading model...")
def load_model(directory):
    model = BartForConditionalGeneration.from_pretrained(directory)
    tokenizer = BartTokenizer.from_pretrained(directory)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return model.to(device), tokenizer, device


finetuned_directory = "../outputs/models/bart_finetuned_samsum_20250510203338"

model, tokenizer, device = load_model(finetuned_directory)
model.eval()

st.title("Dialogue Summarizer with BART")
st.markdown("Enter a dialogue below to get a summary.")

text_input = st.text_area("Enter dialogue here:", height=250)

if st.button("Summarize"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Running model...", show_time=True):
            inputs = tokenizer([text_input], return_tensors="pt", truncation=True, max_length=1024).to(device)
            summary_ids = model.generate(**inputs)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        st.subheader("Summary:")
        st.success(summary)
