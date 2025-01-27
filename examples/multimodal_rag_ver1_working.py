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
import sqlite3
import fitz  # PyMuPDF for PDF processing

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

def load_dataset_cached():
    dataset = load_dataset("davanstrien/ufo-ColPali", split="train")
    return dataset

def init_colpali_model():
    model = ColPali.from_pretrained(
        "davanstrien/finetune_colpali_v1_2-ufo-4bit",
        torch_dtype=torch.bfloat16,
        device_map="mps"  # Adjust based on your hardware
    )
    return model

def init_colpali_processor():
    processor = ColPaliProcessor.from_pretrained(
        "vidore/colpaligemma-3b-pt-448-base"
    )
    return processor

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

def search_similar_images(query_embedding, num_results, qdrant_client, collection_name):
    """Search for similar images in Qdrant"""
    search_result = qdrant_client.query_points(
        collection_name=collection_name,
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

def process_query(query_text, num_results, dataset, colpali_model, colpali_processor, qdrant_client, model_choice, collection_name):
    """Main function to process query and analyze images"""
    # Get query embedding
    query_embedding = get_query_embedding(query_text, colpali_model, colpali_processor)
    
    # Search for similar images
    search_results = search_similar_images(query_embedding, num_results, qdrant_client, collection_name)
    
    images = []
    results = []
    for point in search_results:
        # Get image from dataset
        image = dataset[point.id]["image"]
        images.append(image)
        results.append({
            "image_id": point.id,
            "score": point.score,
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
            if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                analysis_text += content
                # Update the placeholder with the accumulated text
                analysis_placeholder.markdown(analysis_text)
        
        return analysis_text
    else:
        return None

def manage_database(qdrant_client):
    """Function to manage SQLite database"""
    
    # List collections
    collections_response = qdrant_client.get_collections()
    collections = collections_response.collections
    st.write("### Existing Collections")
    for collection in collections:
        st.write(f"- {collection.name}")
    
    # Create new collection
    st.write("### Create New Collection")
    new_collection_name = st.text_input("Enter new collection name:")
    if st.button("Create Collection"):
        if new_collection_name:
            qdrant_client.create_collection(
                collection_name=new_collection_name,
                vectors_config=models.VectorParams(size=128, distance=models.Distance.COSINE),
            )
            st.success(f"Collection '{new_collection_name}' created successfully!")
            st.rerun()
        else:
            st.warning("Please enter a collection name.")

    # Delete collection
    st.write("### Delete a Collection")
    if collections:
        collection_names = [collection.name for collection in collections]
        delete_collection_name = st.selectbox("Select collection to delete", collection_names)
        delete_confirmation = st.checkbox("I confirm to delete this collection.")
        if st.button("Delete Collection"):
            if delete_confirmation:
                qdrant_client.delete_collection(collection_name=delete_collection_name)
                st.success(f"Collection '{delete_collection_name}' deleted successfully!")
                st.rerun()
            else:
                st.warning("Please confirm the deletion by checking the box.")
    else:
        st.write("No collections to delete.")

def add_new_pdf(qdrant_client, colpali_model, colpali_processor):
    """Function to add new PDF documents"""
    st.subheader("Add New PDF Documents")
    
    if 'pdf_converted' not in st.session_state:
        st.session_state['pdf_converted'] = False
        st.session_state['pdf_images'] = []
        st.session_state['pdf_document'] = None
        st.session_state['indexing_in_progress'] = False

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file is not None:
        st.session_state['pdf_document'] = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        if st.button("Convert PDF to Images"):
            # Process the PDF file
            pdf_document = st.session_state['pdf_document']
            images = []
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
            st.session_state['pdf_images'] = images
            st.session_state['pdf_converted'] = True
            # Display the first page of the PDF
            st.image(images[0], caption="First Page of the PDF", use_container_width=True)
            st.success("PDF converted to images.")
    
    if st.session_state['pdf_converted']:
        st.write("### Select Collection for Indexing")
        # Option to create new collection or select existing one
        collection_option = st.radio("Do you want to create a new collection or use an existing one?",
                                     ('Create New Collection', 'Use Existing Collection'), index=1)
        if collection_option == 'Create New Collection':
            new_collection_name = st.text_input("Enter new collection name for PDF embeddings:")
            if st.button("Create New Collection and Index"):
                if new_collection_name:
                    # Create new collection
                    qdrant_client.create_collection(
                        collection_name=new_collection_name,
                        vectors_config=models.VectorParams(size=128, distance=models.Distance.COSINE),
                    )
                    st.success(f"Collection '{new_collection_name}' created successfully!")
                    collection_name = new_collection_name
                    proceed_to_index = True
                else:
                    st.warning("Please enter a collection name.")
                    proceed_to_index = False
            else:
                proceed_to_index = False
        else:
            collections_response = qdrant_client.get_collections()
            collections = collections_response.collections
            if collections:
                collection_names = [col.name for col in collections]
                collection_name = st.selectbox("Select collection to add the PDF to:", collection_names)
                proceed_to_index = st.button("Index into Selected Collection")
            else:
                st.warning("No existing collections found. Please create a new collection.")
                proceed_to_index = False
        if proceed_to_index:
            st.session_state['indexing_in_progress'] = True
            images = st.session_state['pdf_images']
            # Process and encode images
            with st.spinner("Embedding and indexing images..."):
                with torch.no_grad():
                    batch_images = colpali_processor.process_images(images).to(
                        colpali_model.device
                    )
                    image_embeddings = colpali_model(**batch_images)
                
                # Prepare points for Qdrant
                points = []
                for j, embedding in enumerate(image_embeddings):
                    # Convert the embedding to a list of vectors
                    multivector = embedding.cpu().float().numpy().tolist()
                    points.append(
                        models.PointStruct(
                            id=j,  # we just use the index as the ID
                            vector=multivector,  # This is now a list of vectors
                            payload={
                                "source": "uploaded PDF"
                            },  # can also add other metadata/data
                        )
                    )
                qdrant_client.upsert(
                    collection_name=collection_name,
                    points=points,
                    wait=True,
                )
            st.success(f"Images embedded and indexed into the collection '{collection_name}' successfully!")
            st.session_state['indexing_in_progress'] = False
            # Reset the state
            st.session_state['pdf_converted'] = False
            st.session_state['pdf_images'] = []
            st.session_state['pdf_document'] = None

def main():
    st.title("Multimodal RAG for UFO Document")

    # Add model selection
    model_choice = st.radio(
        "Select Vision Model:",
        ("Local (LM Studio)", "Gemini"),
        horizontal=True
    )

    # Sidebar for model/data management and adding new PDFs
    with st.sidebar:
        st.header("Model and Data Management")
        if 'models_loaded' not in st.session_state:
            st.session_state['models_loaded'] = False

        if not st.session_state['models_loaded']:
            if st.button("Load Models and Data"):
                with st.spinner("Loading models and data..."):
                    try:
                        st.session_state['dataset'] = load_dataset_cached()
                        st.session_state['colpali_model'] = init_colpali_model()
                        st.session_state['colpali_processor'] = init_colpali_processor()
                        st.session_state['qdrant_client'] = init_qdrant_client()
                        st.session_state['models_loaded'] = True
                    except Exception as e:
                        st.error(f"Error loading models: {e}")
                if st.session_state['models_loaded']:
                    st.success("Models and data loaded.")
        else:
            if st.button("Unload Models and Data"):
                del st.session_state['dataset']
                del st.session_state['colpali_model']
                del st.session_state['colpali_processor']
                del st.session_state['qdrant_client']
                st.session_state['models_loaded'] = False
                st.success("Models and data unloaded.")

        if st.session_state.get('models_loaded', False):
            st.header("Database Management")
            manage_database(st.session_state['qdrant_client'])
            add_new_pdf(st.session_state['qdrant_client'], st.session_state['colpali_model'], st.session_state['colpali_processor'])
    
    if not st.session_state.get('models_loaded', False):
        st.warning("Please load the models and data to proceed.")
        st.stop()

    if st.session_state.get('indexing_in_progress', False):
        st.warning("Indexing is in progress. Please wait until it completes before entering a query.")
        st.stop()

    # Check if PDF conversion is not pending
    if st.session_state.get('pdf_converted', False):
        st.warning("Please complete the embedding and indexing process before entering a query.")
        st.stop()

    # User inputs
    query_text = st.text_input("Enter your query:")
    num_results = st.number_input("Number of results to display:", min_value=1, max_value=10, value=3)
    
    # Collection selection
    collections_response = st.session_state['qdrant_client'].get_collections()
    collections = collections_response.collections
    collection_names = [col.name for col in collections]
    if collection_names:
        collection_name = st.selectbox("Select collection to query:", collection_names)
    else:
        st.warning("No collections available. Please create or add data to a collection first.")
        st.stop()
    
    # Perform search and analysis when the button is clicked
    if st.button("Submit Query"):
        if not query_text:
            st.warning("Please enter a query.")
        else:
            # Clear previous outputs
            st.session_state['outputs'] = {}
            st.write("Processing your query...")
            try:
                analysis_text = process_query(
                    query_text=query_text,
                    num_results=num_results,
                    dataset=st.session_state['dataset'],
                    colpali_model=st.session_state['colpali_model'],
                    colpali_processor=st.session_state['colpali_processor'],
                    qdrant_client=st.session_state['qdrant_client'],
                    model_choice=model_choice,
                    collection_name=collection_name
                )
            except Exception as e:
                st.error(f"Error during query processing: {e}")
        if 'analysis_text' in locals() and analysis_text:
            st.write("Analysis:")
            st.write(analysis_text)

    if st.button("Clear Output"):
        st.rerun()

if __name__ == "__main__":
    main()