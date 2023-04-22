import os
import json


# def loading_json_chunks(json_chunks_directory: str):
#     for json_file_path in json_files_paths:
#         with open(json_file_path, "r") as f:
#             documents = json.load(f)

#         texts = [doc["page_content"] for doc in documents]


def loading_json_chunks(json_chunks_directory: str):
    all_values = []
    for filename in os.listdir(json_chunks_directory):
        if filename.endswith('.json'):
            with open(os.path.join(json_chunks_directory, filename), 'r') as f:
                data = json.load(f)
                for item in data:
                    all_values.append(item['page_content'])
    return all_values
