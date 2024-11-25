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
    
    return unique_hash

def read_hash_from_file(input_file):
    """
    Read the unique hash from the specified file.
    """
    try:
        with open(input_file, "r") as f:
            # Read the file's contents
            first_line = f.readline().strip()
            if first_line.startswith("Unique hash:"):
                # Extract the hash after "Unique Hash:"
                unique_hash = first_line.split(":", 1)[1].strip()
                print(f"Read hash: {unique_hash}")
                return unique_hash
            else:
                raise ValueError("File does not contain a valid hash format.")
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def write_hash_to_file(output_file, unique_hash):
    """
    Write the unique hash to a file.
    """
    with open(output_file, "w") as f:
        f.write(f"Unique hash: {unique_hash}\n")
    print(f"Hash written to {output_file}")


def compare_hashes(input_file, calculated_hash):
    """
    Compare the hash stored in the file with the calculated hash.
    """
    stored_hash = read_hash_from_file(input_file)
    
    if stored_hash is None:
        print("No valid hash found in the file.")
        write_hash_to_file(input_file, calculated_hash)
        return False  # Unable to compare if no hash is found
    
    if stored_hash == calculated_hash:
        print("Hashes match. The data has not been modified.")
        return True
    else:
        print("Hashes do not match. The data might have been modified.")
        write_hash_to_file(input_file, calculated_hash)
        return False
    

folder_train = "example_data/train_data"
folder_test = "example_data/test_data"
unique_hash_train = get_folder_metadata_with_unique_hash(folder_train)
unique_hash_test = get_folder_metadata_with_unique_hash(folder_test)

compare_hashes("train_hashes.txt",unique_hash_train)
compare_hashes("test_hashes.txt",unique_hash_test)

