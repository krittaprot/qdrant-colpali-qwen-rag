import streamlit as st
from config import BATCH_SIZE, MODEL_PATH, DB_PATH, IMAGE_DIR, EXCLUDE_COLLECTIONS  # local file
from utils.pdf_processor import PDFProcessor  # local file
from utils.qdrant_utils import create_collection, upload_batch  # local file
from colpali_engine.models import ColPali, ColPaliProcessor
from qdrant_client import QdrantClient
from qdrant_client.http import models
import torch
import os
import numpy as np
from tqdm import tqdm
from PIL import Image

# Function to display an image given its filename
def display_image(filename):
    image_path = os.path.join('stored_images', filename)
    try:
        img = Image.open(image_path)
        st.image(img, caption=f'Result Image: {filename}')
    except Exception as e:
        st.error(f"Error loading image {filename}: {e}")

def init_session_state():
    """Initialize session state variables"""
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = None
        st.session_state.model = None
        st.session_state.processor = None
        st.session_state.qdrant_client = None
        st.session_state.pdf_processor = None
        st.session_state.current_images = []
        
@st.cache_resource(show_spinner=False)
def system_initialization():
    """Load and initialize all required artifacts and classes"""
    try:
        model = ColPali.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="mps"
        ).eval()
        processor = ColPaliProcessor.from_pretrained(MODEL_PATH)
        return model, processor
    except Exception as e:
        raise Exception(f"Error loading models: {e}")

def get_qdrant_client():
    """Connect to Qdrant Client"""
    try:
        db_path = os.path.join(os.getcwd(), DB_PATH)
        qdrant_client = QdrantClient(path=db_path)
        return qdrant_client
    except Exception as e:
        raise Exception(f"Error connecting to Qdrant Client: {e}")

def get_collections(client):
    """Get list of all collections"""
    try:
        collections = client.get_collections().collections
        # Filter out excluded collections
        return [collection.name for collection in collections if collection.name not in EXCLUDE_COLLECTIONS]
    except Exception as e:
        st.error(f"Error fetching collections: {e}")
        return []

def delete_collection(client, collection_name):
    """Delete a collection"""
    try:
        client.delete_collection(collection_name=collection_name)
        return True
    except Exception as e:
        st.error(f"Error deleting collection: {e}")
        return False

def get_patches(image_size, model_processor, model, model_name):
    if model_name == "colPali":
        return model_processor.get_n_patches(image_size,
                                             patch_size=model.patch_size)
    elif model_name == "colQwen":
        return model_processor.get_n_patches(image_size,
                                             patch_size=model.patch_size,
                                             spatial_merge_size=model.spatial_merge_size)
    return None, None

def embed_and_mean_pool_batch(image_batch, model_processor, model, model_name):
    with torch.no_grad():
        processed_images = model_processor.process_images(image_batch).to(model.device)
        image_embeddings = model(**processed_images)
    image_embeddings_batch = image_embeddings.cpu().float().numpy().tolist()
    pooled_by_rows_batch = []
    pooled_by_columns_batch = []
    for image_embedding, tokenized_image, image in zip(image_embeddings,
                                                       processed_images.input_ids,
                                                       image_batch):
        x_patches, y_patches = get_patches(image.size, model_processor, model, model_name)
        image_tokens_mask = (tokenized_image == model_processor.image_token_id)
        image_tokens = image_embedding[image_tokens_mask].view(x_patches, y_patches, model.dim)
        pooled_by_rows = torch.mean(image_tokens, dim=0)
        pooled_by_columns = torch.mean(image_tokens, dim=1)
        image_token_idxs = torch.nonzero(image_tokens_mask.int(), as_tuple=False)
        first_image_token_idx = image_token_idxs[0].cpu().item()
        last_image_token_idx = image_token_idxs[-1].cpu().item()
        prefix_tokens = image_embedding[:first_image_token_idx]
        postfix_tokens = image_embedding[last_image_token_idx + 1:]
        pooled_by_rows = torch.cat((prefix_tokens, pooled_by_rows, postfix_tokens), dim=0).cpu().float().numpy().tolist()
        pooled_by_columns = torch.cat((prefix_tokens, pooled_by_columns, postfix_tokens), dim=0).cpu().float().numpy().tolist()
        pooled_by_rows_batch.append(pooled_by_rows)
        pooled_by_columns_batch.append(pooled_by_columns)
    return image_embeddings_batch, pooled_by_rows_batch, pooled_by_columns_batch

def batch_embed_query(query_batch, model_processor, model):
    with torch.no_grad():
        processed_queries = model_processor.process_queries(query_batch).to(model.device)
        query_embeddings_batch = model(**processed_queries)
    return query_embeddings_batch.cpu().float().numpy()

def reranking_search_batch(query_batch,
                           collection_name,
                           search_limit=20,
                           prefetch_limit=200):
    search_queries = [
        models.QueryRequest(
            query=query,
            prefetch=[
                models.Prefetch(
                    query=query,
                    limit=prefetch_limit,
                    using="mean_pooling_columns"
                ),
                models.Prefetch(
                    query=query,
                    limit=prefetch_limit,
                    using="mean_pooling_rows"
                ),
            ],
            limit=search_limit,
            with_payload=True,
            with_vector=False,
            using="original"
        ) for query in query_batch
    ]
    return st.session_state.qdrant_client.query_batch_points(
        collection_name=collection_name,
        requests=search_queries
    )

def main():
    st.title("Document Search with colPali")
    init_session_state()

    # Load models automatically on startup
    if not st.session_state.models_loaded:
        with st.spinner("Initializing the system..."):
            st.session_state.model, st.session_state.processor = system_initialization()
            st.session_state.qdrant_client = get_qdrant_client()
            st.session_state.pdf_processor = PDFProcessor()
            if st.session_state.model:
                st.session_state.models_loaded = True
            st.success("The system loaded successfully!")

    # Sidebar content
    st.sidebar.header("PDF Upload")
    uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")
    if uploaded_file:
        with st.spinner("Loading..."):
            # Store images and get their metadata
            stored_images, metadata, identical_flag = st.session_state.pdf_processor.convert_and_store_pdf(uploaded_file)
            st.session_state.current_images = stored_images
            
            if identical_flag:
                st.success("Uploaded PDF is identical to saved files.")
            else:
                st.success(f"Converted and stored {len(st.session_state.current_images)} pages")
             
            # Collection management
            collection_name = st.text_input("Collection name")
            if collection_name:
                if collection_name in get_collections(st.session_state.qdrant_client):
                    st.warning(f"Collection '{collection_name}' already exists!")
                elif st.button("Create Collection and Index"):
                    try:
                        create_collection(st.session_state.qdrant_client, collection_name)
                        dataset_source = collection_name  # Adjust as needed
                        with tqdm(total=len(stored_images), desc=f"Uploading progress of \"{collection_name}\" collection") as pbar:
                            for i in range(0, len(stored_images), BATCH_SIZE):
                                image_batch = stored_images[i:i + BATCH_SIZE]
                                current_batch_size = len(image_batch)
                                try:
                                    original_batch, pooled_by_rows_batch, pooled_by_columns_batch = embed_and_mean_pool_batch(
                                        image_batch,
                                        st.session_state.processor,
                                        st.session_state.model,
                                        "colPali"
                                    )
                                except Exception as e:
                                    print(f"Error during embed: {e}")
                                    continue
                                try:
                                    # Create payload_batch
                                    payload_batch = [
                                        {
                                            "source": dataset_source,
                                            "index": j
                                        }
                                        for j in range(i, i + current_batch_size)
                                    ]
                                    # Call upload_batch with payload_batch
                                    upload_batch(
                                        st.session_state.qdrant_client,
                                        collection_name,
                                        np.asarray(original_batch, dtype=np.float32),
                                        np.asarray(pooled_by_rows_batch, dtype=np.float32),
                                        np.asarray(pooled_by_columns_batch, dtype=np.float32),
                                        payload_batch
                                    )
                                except Exception as e:
                                    print(f"Error during upsert: {e}")
                                    continue
                                # Update the progress bar
                                pbar.update(current_batch_size)
                        st.success("Indexing complete!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

    # Collection Management Section in Sidebar
    st.sidebar.markdown("### Collection Management")
    collections = get_collections(st.session_state.qdrant_client)
    if collections:
        st.sidebar.write("Existing Collections:")
        for collection in collections:
            col1, col2 = st.sidebar.columns([3, 1])
            with col1:
                st.write(collection)
            with col2:
                if st.button("Delete", key=f"delete_{collection}"):
                    if delete_collection(st.session_state.qdrant_client, collection):
                        st.sidebar.success(f"Deleted collection: {collection}")
                        st.rerun()
    else:
        st.sidebar.write("No collections found")

    # Query box and search functionality
    query = st.text_input("Enter your query")
    collection_name = st.selectbox("Select Collection", get_collections(st.session_state.qdrant_client))
    top_k = st.number_input("Number of top results to display", min_value=1, value=2)
    if st.button("Search"):
        if query:
            colpali_query = batch_embed_query([query], st.session_state.processor, st.session_state.model)
            results = reranking_search_batch(colpali_query, collection_name, search_limit=top_k)
            for result in results:
                for point_idx, point in enumerate(result.points):
                    score = point.score
                    index = point.payload.get('index')  # Extract the index
                    st.write(f"Result {point_idx + 1}, Index: {index}, Relevance Score: {score:.2f}")
                    stored_images[index]

if __name__ == "__main__":
    main()