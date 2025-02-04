import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
import fitz  # PyMuPDF for PDF reading

# Initialize the Groq client with your API key
os.environ["GROQ_API_KEY"] = "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"
client = Groq()

# Initialize the SentenceTransformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract text from PDFs
def extract_text_from_pdfs(pdf_folder):
    tracks = []
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            # Extract program name from the filename (excluding extension)
            program_name = os.path.splitext(pdf_file)[0]
            
            # Read PDF content
            doc = fitz.open(pdf_path)
            content = ""
            for page in doc:
                content += page.get_text("text")
            
            # Store the program name and guidelines as a track
            tracks.append({"name": program_name, "description": content.strip()})
    
    return tracks

# Create a vector database (FAISS index) from track descriptions
def create_vector_db(tracks):
    descriptions = [track['description'] for track in tracks]
    embeddings = model.encode(descriptions)  # Create embeddings for descriptions
    
    # Normalize embeddings
    embeddings = np.array(embeddings)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Create a FAISS index for similarity search
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    return index, tracks

# Function to find the best matching track from the vector DB
def find_best_match(user_brief, index, tracks):
    # Convert the user's brief into an embedding
    user_embedding = model.encode([user_brief])
    
    # Normalize the user's embedding
    user_embedding = user_embedding / np.linalg.norm(user_embedding)
    
    # Search for the closest match in the FAISS index
    D, I = index.search(user_embedding, k=1)  # Get the top 1 most similar track
    
    # Return the most similar track
    best_match = tracks[I[0][0]]
    return best_match

# Function to query the LLM (Groq) for eligibility with a 100-word brief and reason for eligibility
def query_llm_for_eligibility(track_name, track_description, user_brief):
    # Prepare the prompt for Groq's completion
    prompt = f"""
    You are an AI advisor. Here is a track named "{track_name}" with the description:

    {track_description}

    Based on the following user brief:

    {user_brief}

    Give a 100-word brief about the program and explain why this user is eligible for this track.
    """

    # Make the API call to Groq
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # The model to use, modify as needed
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        top_p=1,
        stream=True
    )

    # Collect the response from the streamed completion
    eligibility_info = ""
    for chunk in completion:
        eligibility_info += chunk.choices[0].delta.content or ""

    # Truncate the response to 100 words
    eligibility_info_words = eligibility_info.split()
    eligibility_info = ' '.join(eligibility_info_words[:100])  # Keep only the first 100 words

    return eligibility_info.strip()

# Main function
def main():
    user_brief = input("Enter your startup brief: ")
    
    # Extract tracks from PDFs in a folder
    pdf_folder = "V3/data"  # Set the correct path to your PDFs folder
    tracks = extract_text_from_pdfs(pdf_folder)
    
    # Create the vector DB from tracks
    index, tracks = create_vector_db(tracks)
    
    # Find the best match for the user's brief in the vector DB
    best_match = find_best_match(user_brief, index, tracks)
    
    # Output the closest track based on the user's input
    print("\n===============================")
    print(f"Best Matching Track: {best_match['name']}")
    print("===============================")
    print(f"Track Description:\n{best_match['description']}")
    
    # Query the Groq API for eligibility information
    eligibility_info = query_llm_for_eligibility(best_match['name'], best_match['description'], user_brief)
    print("\nEligibility Brief (100 words):")
    print(f"{eligibility_info}")

if __name__ == "__main__":
    main()
