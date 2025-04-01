import os
import requests
import json
from pymilvus import connections, Collection, FieldSchema, DataType, CollectionSchema, utility
import torch

def store_vector_in_milvus(vector):
    """Stores the vector in the Milvus database.

    Args:
        vector (torch.Tensor): The vector to store.
    """
    # TODO: Install pymilvus: pip install pymilvus

    # Get Milvus connection details from environment variables
    milvus_uri = "http://47.130.66.67:19530" #os.environ.get("MILVUS_URI")
    milvus_db_name = "default" #os.environ.get("MILVUS_DB_NAME")
    milvus_user = "" #os.environ.get("MILVUS_USER")
    milvus_password = "" #os.environ.get("MILVUS_PASSWORD")

    # Connect to Milvus
    try:
        connections.connect(
            uri=milvus_uri,
            db_name=milvus_db_name,
            user=milvus_user,
            password=milvus_password,
        )
        print("Connected to Milvus successfully!")
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        return

    # Define collection name
    collection_name = "fashion_clip_vectors"

    # Check if the collection exists
    if utility.has_collection(collection_name):
        collection = Collection(collection_name)
        print(f"Collection '{collection_name}' already exists.")
    else:
        # Define fields for the collection
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512)  # CLIP vector dimension is 512
        ]
        schema = CollectionSchema(fields=fields, description="FashionCLIP embeddings")

        # Create the collection
        collection = Collection(collection_name, schema=schema)
        print(f"Collection '{collection_name}' created successfully!")

        # Create an index for the vector field
        index_params = {"metric_type": "IP", "index_type": "IVF_FLAT", "params": {"nlist": 1024}}
        collection.create_index(field_name="embedding", index_params=index_params)
        print(f"Index created for field 'embedding' successfully!")

    # Convert the vector to a list of lists
    insert_data = [vector.tolist()]

    # Insert the vector into the collection
    try:
        collection.insert(insert_data)
        collection.flush()
        print("Vector inserted successfully!")
    except Exception as e:
        print(f"Failed to insert vector: {e}")

def main():
    """Main function to orchestrate the process."""
    clip_api_url = "http://localhost:5000/get_vector"  # Replace with your API endpoint
    image_url = "https://images.unsplash.com/photo-1661961112951-f2bfd1f253ce?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1472&q=80"
    text = "a black leather jacket"

    try:
        payload = {'image_url': image_url, 'text': text}
        headers = {'Content-type': 'application/json'}
        response = requests.post(clip_api_url, data=json.dumps(payload), headers=headers)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        vector_data = response.json()
        vector = torch.tensor(vector_data['vector'])
        store_vector_in_milvus(vector)

    except requests.exceptions.RequestException as e:
        print(f"Error calling CLIP API: {e}")
    except Exception as e:
        print(f"Error processing response: {e}")

if __name__ == "__main__":
    main()
