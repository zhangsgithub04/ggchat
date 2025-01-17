import streamlit as st
import torch
from transformers import BartTokenizer, BartForConditionalGeneration, pipeline

# Load pre-trained BART model and tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

# Load zero-shot classification pipeline
zero_shot_classifier = pipeline("zero-shot-classification")

def zero_shot_prediction(premise, hypothesis, labels):
    # Use zero-shot classifier to predict relationship
    prediction = zero_shot_classifier(premise, labels)
    return prediction

def bart_generation(premise, hypothesis):
    # Preprocess input text
    inputs = tokenizer(premise, hypothesis, return_tensors='pt')

    # Generate text using BART
    outputs = model.generate(**inputs)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text

def main():
    st.title("Zero-Shot BART App")

    # Add instructions or context upfront
    st.markdown("**Instructions:**")
    st.markdown("* Enter a premise and hypothesis to predict their relationship.")

    premise = st.text_input("Enter premise:")
    hypothesis = st.text_input("Enter hypothesis:")

    labels = ["contradiction", "entailment", "neutral"]

    if st.button("Predict"):
        prediction = zero_shot_prediction(premise, hypothesis, labels)
        st.write("Predicted label:", prediction["labels"][0])

        generated_text = bart_generation(premise, hypothesis)
        st.write("Generated text:", generated_text)

if __name__ == "__main__":
    main()
