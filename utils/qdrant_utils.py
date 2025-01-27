# utils/qdrant_utils.py
from qdrant_client.http import models
import uuid

def create_collection(client, collection_name):
    """Create a new Qdrant collection with the specified configuration"""
    return client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "original": models.VectorParams(
                size=128,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
                hnsw_config=models.HnswConfigDiff(m=0)
            ),
            "mean_pooling_columns": models.VectorParams(
                size=128,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                )
            ),
            "mean_pooling_rows": models.VectorParams(
                size=128,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                )
            )
        }
    )

def upload_batch(client, collection_name, original_batch, pooled_by_rows_batch, 
                pooled_by_columns_batch, payload_batch):
    """Upload a batch of embeddings to Qdrant"""
    try:
        client.upload_collection(
            collection_name=collection_name,
            vectors={
                "mean_pooling_columns": pooled_by_columns_batch,
                "original": original_batch,
                "mean_pooling_rows": pooled_by_rows_batch
            },
            payload=payload_batch,
            ids=[str(uuid.uuid4()) for _ in range(len(original_batch))]
        )
    except Exception as e:
        raise Exception(f"Error during upload: {e}")