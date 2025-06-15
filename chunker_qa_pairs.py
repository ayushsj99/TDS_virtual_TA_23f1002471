import json
import os
import tiktoken
from typing import List, Dict
import re


class QAChunker:
    def __init__(self, model_name: str = "cl100k_base", max_tokens: int = 3500):
        self.tokenizer = tiktoken.get_encoding(model_name)
        self.max_tokens = max_tokens

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def chunk_text(self, text: str, prefix_tokens: int = 0) -> List[str]:
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if not sentence.endswith('.'):
                sentence += '.'
            trial = current_chunk + ' ' + sentence if current_chunk else sentence
            token_count = self.count_tokens(trial) + prefix_tokens

            if token_count > self.max_tokens:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk = trial

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def chunk_qa_pairs(self, qa_pairs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        chunked_qa_pairs = []
        count_kept = 0
        count_chunked = 0
        count_skipped = 0
        total = len(qa_pairs)

        for pair in qa_pairs:
            question = pair["question"]
            answer_list = pair.get("answers", [])

            if not answer_list:
                chunked_qa_pairs.append({
                    "source": "discourse",
                    "question": question,
                    "answer": None,
                    "question_url": pair.get("question_url", ""),
                    "question_id": pair.get("question_id"),
                    "topic_id": pair.get("topic_id")
                })
                count_skipped += 1
                continue

            for ans in answer_list:
                answer = ans.get("text", "")
                if not answer.strip():
                    continue

                prefix_tokens = self.count_tokens(question)
                full_tokens = self.count_tokens(question + " " + answer)

                if full_tokens <= self.max_tokens:
                    chunked_qa_pairs.append({
                        "source": "discourse",
                        "question": question,
                        "answer": answer,
                        "question_url": pair.get("question_url", ""),
                        "answered_by": ans.get("answered_by", ""),
                        "answer_url": ans.get("url", ""),
                        "question_id": pair.get("question_id"),
                        "topic_id": pair.get("topic_id")
                    })
                    count_kept += 1
                else:
                    answer_chunks = self.chunk_text(answer, prefix_tokens=prefix_tokens)
                    for chunk in answer_chunks:
                        chunked_qa_pairs.append({
                            "source": "discourse",
                            "question": question,
                            "answer": chunk,
                            "question_url": pair.get("question_url", ""),
                            "answered_by": ans.get("answered_by", ""),
                            "answer_url": ans.get("url", ""),
                            "question_id": pair.get("question_id"),
                            "topic_id": pair.get("topic_id")
                        })
                    count_chunked += 1

        print(f"\U0001F50D Total QA pairs processed: {total}")
        print(f"❌ Skipped (no answers): {count_skipped}")
        print(f"✅ Kept as-is (short enough): {count_kept}")
        print(f"✂️ Chunked (too long): {count_chunked}")

        return chunked_qa_pairs


    def chunk_markdown_files(self, md_folder: str, chunk_size: int = 1200, overlap: int = 100) -> List[Dict[str, str]]:
        md_chunks = []

        for filename in os.listdir(md_folder):
            if not filename.endswith(".md"):
                continue

            filepath = os.path.join(md_folder, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            # Extract metadata using regex
            meta_match = re.search(r"---\s*(.*?)\s*---", content, re.DOTALL)
            metadata = {}
            if meta_match:
                meta_content = meta_match.group(1)
                for line in meta_content.splitlines():
                    if ":" in line:
                        key, value = line.split(":", 1)
                        metadata[key.strip()] = value.strip().strip('"')

                # Remove the metadata block from the content
                content = content.replace(meta_match.group(0), '').strip()
            
            content = re.sub(r'^\s*[-=*]{3,}\s*$', '', content, flags=re.MULTILINE)

            title = metadata.get("title", "")
            original_url = metadata.get("original_url", "")
            downloaded_at = metadata.get("downloaded_at", "")

            tokens = self.tokenizer.encode(content)
            i = 0
            while i < len(tokens):
                chunk_tokens = tokens[i:i + chunk_size]
                chunk_text = self.tokenizer.decode(chunk_tokens)

                md_chunks.append({
                    "source": "markdown",
                    "source_file": filename,
                    "chunk_text": chunk_text,
                    "title": title,
                    "original_url": original_url,
                    "downloaded_at": downloaded_at
                })

                i += chunk_size - overlap  # Slide window with overlap

        print(f"📚 Chunked markdown files from '{md_folder}' into {len(md_chunks)} chunks.")
        return md_chunks



def load_jsonl(file_path: str) -> List[Dict[str, str]]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(data: List[Dict[str, str]], file_path: str):
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    qa_input_path = "discourse_qa_pairs.jsonl"
    md_input_folder = "tds_pages_md"
    combined_output_path = "chunked_qa_pairs.jsonl"

    discourse_qa_pairs = load_jsonl(qa_input_path)
    chunker = QAChunker(max_tokens=3500)

    # Process QA pairs
    chunked_qa = chunker.chunk_qa_pairs(discourse_qa_pairs)

    # Process Markdown files
    markdown_chunks = chunker.chunk_markdown_files(md_input_folder, chunk_size=1200, overlap=100)

    # Combine and save
    all_data = chunked_qa + markdown_chunks
    save_jsonl(all_data, combined_output_path)

    print(f"✅ Saved total {len(all_data)} chunks to {combined_output_path}")
