import os
import base64
from io import BytesIO
import streamlit as st
import torch
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http import models
from datasets import load_dataset
from colpali_engine.models import ColPali, ColPaliProcessor
from openai import OpenAI
from dotenv import load_dotenv

# Set environment variables
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
load_dotenv()

# Create a function to get the appropriate client based on the selected model
def get_client(model_choice):
    if model_choice == "Local (LM Studio)":
        return OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    else:  # Gemini
        return OpenAI(
            api_key=os.getenv('GOOGLE_API_KEY'),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

@st.cache_resource
def load_dataset_cached():
    dataset = load_dataset("davanstrien/ufo-ColPali", split="train")
    return dataset

@st.cache_resource
def init_colpali_model():
    model = ColPali.from_pretrained(
        "davanstrien/finetune_colpali_v1_2-ufo-4bit",
        torch_dtype=torch.bfloat16,
        device_map="mps"  # Adjust based on your hardware
    )
    return model

@st.cache_resource
def init_colpali_processor():
    processor = ColPaliProcessor.from_pretrained(
        "vidore/colpaligemma-3b-pt-448-base"
    )
    return processor

@st.cache_resource
def init_qdrant_client():
    db_path = os.path.join(os.getcwd(), "qdrant_local_db")
    qdrant_client = QdrantClient(path=db_path)
    return qdrant_client

def get_query_embedding(query_text, colpali_model, colpali_processor):
    """Generate embedding for the query text"""
    colpali_model.eval()
    with torch.no_grad():
        batch_query = colpali_processor.process_queries([query_text]).to(
            colpali_model.device
        )
        query_embedding = colpali_model(**batch_query)
    return query_embedding[0].cpu().float().numpy().tolist()

def search_similar_images(query_embedding, num_results, qdrant_client):
    """Search for similar images in Qdrant"""
    search_result = qdrant_client.query_points(
        collection_name="ufo-binary",
        query=query_embedding,
        limit=num_results,
        timeout=100,
        search_params=models.SearchParams(
            quantization=models.QuantizationSearchParams(
                ignore=False,
                rescore=True,
                oversampling=2.0,
            )
        )
    )
    return search_result

def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def analyze_images_with_vision_api(images, question, model_choice):
    """Analyze images using OpenAI Vision API"""
    # Initialize the OpenAI client
    client = get_client(model_choice)
    
    # Create a more specific prompt
    prompt = f"""Based on the following query: '{question}'
    Please analyze the provided image(s) and provide an answer to the query with supporting facts or findings (if any).
    Focus on answering the query first, then briefly mention relevant details from the images if necessary.
    Be concise and specific."""

    # Build the message content with the structured prompt and images
    message_content = [{"type": "text", "text": prompt}]
    
    for image in images:
        # Convert PIL image to base64
        encoded_image = image_to_base64(image)
        # Append the image to the message content
        message_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_image}",
            },
        })
    
    try:
        # Create the completion request
        response = client.chat.completions.create(
            model= "qwen2-vl-7b-instruct" if model_choice == "Local (LM Studio)" else "gemini-2.0-flash-exp",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant analyzing UFO documents. Focus on answering queries directly and precisely."
                },
                {
                    "role": "user",
                    "content": message_content,
                }
            ],
            max_tokens=300,
            stream=True
        )
        
        return response
        
    except Exception as e:
        st.error(f"Error in vision API: {str(e)}")
        return None

def process_query(query_text, num_results, dataset, colpali_model, colpali_processor, qdrant_client, model_choice):
    """Main function to process query and analyze images"""
    # Get query embedding
    query_embedding = get_query_embedding(query_text, colpali_model, colpali_processor)
    
    # Search for similar images
    search_results = search_similar_images(query_embedding, num_results, qdrant_client)
    
    images = []
    results = []
    for point in search_results.points:  # Changed from search_results to search_results.points
        # Get image from dataset
        image = dataset[point.id]["image"]  # Changed from result.id to point.id
        images.append(image)
        results.append({
            "image_id": point.id,  # Changed from result.id to point.id
            "score": point.score,  # Changed from result.score to point.score
        })
    
    # Display images with relevance scores horizontally
    st.subheader("Search Results")
    cols = st.columns(len(results))
    for i, (col, result) in enumerate(zip(cols, results)):
        with col:
            image = dataset[result['image_id']]['image']
            st.image(image, caption=f"Score: {round(result['score'], 2)}", use_container_width=True)

    
    # Analyze images with VLM
    st.write("Generating answer...")
    vision_response = analyze_images_with_vision_api(images, query_text, model_choice)
    
    if vision_response:
        # Process streamed response
        analysis_text = ""
        analysis_placeholder = st.empty()
        for chunk in vision_response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                analysis_text += content
                # Update the placeholder with the accumulated text
                analysis_placeholder.markdown(analysis_text)
        
        return analysis_text
    else:
        return None

def main():
    st.title("Multimodal RAG for UFO Document")

    # Add model selection
    model_choice = st.radio(
        "Select Vision Model:",
        ("Local (LM Studio)", "Gemini"),
        horizontal=True
    )

    # Load models and data
    with st.spinner("Loading models and data..."):
        dataset = load_dataset_cached()
        colpali_model = init_colpali_model()
        colpali_processor = init_colpali_processor()
        qdrant_client = init_qdrant_client()
    st.success("Models and data loaded.")
    
    # User inputs
    query_text = st.text_input("Enter your query:")
    num_results = st.number_input("Number of results to display:", min_value=1, max_value=10, value=3)
    
    # Perform search and analysis when the button is clicked
    if st.button("Submit Query"):
        if not query_text:
            st.warning("Please enter a query.")
        else:
            # Clear previous outputs
            st.session_state['outputs'] = {}
            st.write("Processing your query...")
            analysis_text = process_query(
                query_text=query_text,
                num_results=num_results,
                dataset=dataset,
                colpali_model=colpali_model,
                colpali_processor=colpali_processor,
                qdrant_client=qdrant_client,
                model_choice=model_choice
            )
                
    if st.button("Clear Output"):
        st.rerun()

if __name__ == "__main__":
    main()