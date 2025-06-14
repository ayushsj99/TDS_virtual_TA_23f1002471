from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import glob
import json
from tqdm import tqdm

def get_chunks_with_langchain(text, chunk_size=500, overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    return splitter.split_text(text)

def get_markdown_chunks(folder_paths, chunk_size=500, overlap=50):
    all_chunks = []
    for folder_path in folder_paths:
        md_files = glob.glob(os.path.join(folder_path, "*.md"))
        for filepath in tqdm(md_files, desc=f"Chunking files in {folder_path}"):
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            file_chunks = get_chunks_with_langchain(content, chunk_size, overlap)
            file_id = os.path.splitext(os.path.basename(filepath))[0]
            for i, chunk in enumerate(file_chunks):
                all_chunks.append({
                    "file": file_id,
                    "chunk_id": f"{file_id}_{i}",
                    "text": chunk.strip()
                })
    return all_chunks

if __name__ == "__main__":
    folders = ["discourse_md", "tds_pages_md"]
    chunks = get_markdown_chunks(folders, chunk_size=500, overlap=50)

    with open("combined_chunks.jsonl", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\n✅ Created {len(chunks)} sentence-aware overlapping chunks for embedding.")
