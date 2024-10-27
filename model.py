from transformers import pipeline

# Initialize the models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
text_generator = pipeline("text-generation", model="gpt2")  # Use a suitable model for generation

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

def explain_term(term):
    """
    Explain a complex legal term in simple language.
    """
    explanations = {
        "contract": "A contract is a legal agreement between two or more parties.",
        "tort": "A tort is a wrongful act that causes harm to someone, which can lead to legal liability.",
        "plaintiff": "The plaintiff is the person who brings a case against another in a court of law.",
        "defendant": "The defendant is the person accused of a crime or being sued in a court case.",
    }
    return explanations.get(term.lower(), "Explanation not available for this term.")

def generate_legal_document(prompt, max_length=200):
    """
    Generate a legal document based on a prompt.
    """
    if not prompt or not isinstance(prompt, str):
        return "No prompt provided for document generation."
    
    print("Generating legal document...")
    try:
        generated_text = text_generator(prompt, max_length=max_length, num_return_sequences=1)
        return generated_text[0]['generated_text']
    except Exception as e:
        return f"An error occurred during document generation: {str(e)}"

def answer_question(data, question):
    if 'text' not in data.columns:
        return "No 'text' column found for context in data."
    
    context = " ".join(data['text'].dropna().tolist())
    if not context:
        return "No context available in 'text' column to answer the question."

    print("Answering question...")
    response = qa_model(question=question, context=context)
    return response['answer']

# Example usage
if __name__ == "__main__":
    example_prompt = "Draft a simple rental agreement between a landlord and a tenant."
    print(generate_legal_document(example_prompt))
