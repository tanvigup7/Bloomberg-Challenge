# main.py
import pandas as pd
import json
import PyPDF2  
import torch  # Import torch to check for GPU availability
from model import summarize_text, answer_question
from src.redact import redact_sensitive_info
from src.filters import apply_filters
from src.adversarial_test import test_adversarial_examples
import config
from src.utils import visualize_word_cloud
from src.utils import visualize_word_cloud, visualize_word_frequency, visualize_sentence_length




def main():
    import PyPDF2
    device = 0 if torch.cuda.is_available() else -1  # Automatically select GPU if available

from transformers import pipeline

    # Use the GPU if available
device = 0  # Use 0 for GPU, or -1 for CPU



# Initialize the models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
text_generator = pipeline("text-generation", model="gpt2")


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.
    """
    text = ""
    with PyPDF2.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"  # Extract text from each page
    return text

def summarize_text(text):
    """
    Generate a summary of the provided text.
    """
    if not text or not isinstance(text, str):
        return "No text available for summarization."
    
    print("Generating summary...")
    max_input_length = 1024
    if len(text) > max_input_length:
        text = text[:max_input_length]  # Truncate text to max length

    try:
        summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"An error occurred during summarization: {str(e)}"

# Example usage
if __name__ == "__main__":
    pdf_path = "2406.17186v2.pdf"  # Update this path to your downloaded PDF file
    extracted_text = extract_text_from_pdf(pdf_path)
    data = pd.DataFrame({'text': [extracted_text]})  # Create a DataFrame with the extracted text

    summary = summarize_text(extracted_text)
    print("Summary of the document:")
    print(summary)


    print("Redacting sensitive information...")
    redacted_data = redact_sensitive_info(data)
    print("Sensitive information redacted.")

    print("Applying filters to data...")
    filtered_data = apply_filters(redacted_data)
    print("Filters applied.")

    print("Visualizing data...")
    visualize_word_cloud(filtered_data, column="text")
    visualize_word_frequency(filtered_data, column="text")
    visualize_sentence_length(filtered_data, column="text")

    if 'text' in filtered_data.columns:
        print("Summarizing text...")
        full_text = " ".join(filtered_data['text'].dropna().tolist())
        summary = summarize_text(full_text)
        print("Summary:", summary)
    else:
        print("No 'text' column found for summarization.")

    if 'text' in filtered_data.columns:
        print("Answering questions...")
        question = "What is the main topic of the data?"
        answer = answer_question(filtered_data, question)
        print("Answer:", answer)
    else:
        print("No 'text' column found for question answering.")

    print("Running adversarial tests...")
    test_adversarial_examples(filtered_data, column="text")

if __name__ == "__main__":
    main()
