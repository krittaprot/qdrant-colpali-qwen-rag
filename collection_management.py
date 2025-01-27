import streamlit as st
from utils.qdrant_utils import delete_collection
from app import get_collections, get_qdrant_client

def main():
    st.title("Collection Management")

    # Get existing collections
    collections = get_collections(get_qdrant_client())

    if collections:
        st.write("Existing Collections:")
        for collection in collections:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(collection)
            with col2:
                if st.button("Delete", key=f"delete_{collection}"):
                    if delete_collection(get_qdrant_client(), collection):
                        st.success(f"Deleted collection: {collection}")
                        st.experimental_rerun()  # Rerun the app to refresh the list
    else:
        st.write("No collections found")

if __name__ == "__main__":
    main()