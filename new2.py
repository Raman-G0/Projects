import json
import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# === Load JSON Data ===
def load_legal_data(json_path):
    with open(json_path, "r", encoding="utf-8") as file:
        return json.load(file)

# === Load AI Model for Text Embeddings ===
def load_embedding_model():
    """Loads SentenceTransformer model for embeddings."""
    return SentenceTransformer("all-MiniLM-L6-v2")

# === Load Free LLM Model (Mistral-7B) ===
def load_llm():
    """Loads an open-source model (Mistral-7B) for answering queries."""
    return pipeline("text-generation", model="mistralai/Mistral-7B-Instruct", device="cpu")

# === Create Embeddings for FAISS ===
def create_embeddings(data, model):
    """Creates embeddings for legal text sections."""
    texts = [entry["title"] + " " + entry["content"] for entry in data]
    embeddings = np.array([model.encode(text, convert_to_tensor=True).cpu().numpy() for text in texts], dtype="float32")
    return embeddings, texts

# === Build FAISS Index ===
def build_faiss_index(embeddings):
    """Builds a FAISS index for fast search."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# === Search FAISS for Relevant Sections ===
def search_legal_query(query, index, embeddings, texts, model, top_k=3):
    """Finds relevant legal sections using FAISS."""
    query_embedding = model.encode(query, convert_to_tensor=True).cpu().numpy()
    distances, indices = index.search(np.array([query_embedding]), top_k)
    return [texts[idx] for idx in indices[0]]

# === Generate AI Response using Mistral-7B ===
def generate_ai_response(query, legal_text, llm):
    """Uses Mistral-7B (free model) to generate a legal response based on relevant sections."""
    prompt = f"""
    You are a legal assistant. Answer the user's question based on the following legal text:
    {legal_text}

    Question: {query}
    Answer:
    """
    response = llm(prompt, max_length=300, num_return_sequences=1)
    return response[0]["generated_text"]

# === Streamlit Web App ===
def main():
    st.title("⚖️ ChatJury - AI Legal Assistant (Free LLM) 🤖")
    st.subheader("Ask legal questions and get AI-generated answers!")

    # Load legal data
    json_path = r"C:\Users\raman\Downloads\fixed_motor_vehicle_act.json"
    data = load_legal_data(json_path)

    # Load AI models
    embedding_model = load_embedding_model()
    llm = load_llm()

    # Generate embeddings & build FAISS index
    embeddings, texts = create_embeddings(data, embedding_model)
    index = build_faiss_index(embeddings)

    # User input
    query = st.text_input("Enter your legal question:", "")

    if query:
        st.write("🔍 Searching legal database...")
        relevant_sections = search_legal_query(query, index, embeddings, texts, embedding_model)

        # Combine relevant legal text for AI
        legal_text = "\n\n".join(relevant_sections)

        # Generate AI response using Mistral-7B
        ai_response = generate_ai_response(query, legal_text, llm)

        # Display results
        st.write("📜 **Relevant Legal Sections:**")
        for section in relevant_sections:
            st.markdown(f"🔹 {section[:500]}...")  # Show first 500 chars

        st.write("🤖 **AI Answer:**")
        st.success(ai_response)

# Run the Streamlit app
if __name__ == "__main__":
    main()
