import os
import hashlib
import json

def hash_metadata(metadata_list):
    """
    Create a unique hash for the aggregated metadata of all files.
    """
    # Sort metadata list by file path for consistent order
    sorted_metadata = sorted(metadata_list, key=lambda x: x["file_path"])
    # Serialize metadata list
    metadata_json = json.dumps(sorted_metadata, sort_keys=True)
    # Compute SHA-256 hash
    return hashlib.sha256(metadata_json.encode()).hexdigest()

def get_folder_metadata_with_unique_hash(folder_path):
    """
    Collect metadata for all files in the folder and its subfolders, 
    and compute a unique hash for all the metadata.
    """
    metadata_list = []

    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            file_stats = os.stat(file_path)

            # Collect metadata
            metadata = {
                "file_name": file_name,
                "file_path": file_path,
                "size_bytes": file_stats.st_size,
                "creation_time": file_stats.st_ctime,
                "last_modified_time": file_stats.st_mtime,
                "last_accessed_time": file_stats.st_atime
            }
            metadata_list.append(metadata)

    # Generate a unique hash for all metadata
    unique_hash = hash_metadata(metadata_list)
    
    return metadata_list, unique_hash


folder = "example_data"
metadata_list, unique_hash = get_folder_metadata_with_unique_hash(folder)

print(metadata_list)
print(unique_hash)