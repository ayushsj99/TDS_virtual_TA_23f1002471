import os
import re
import glob
import json
from tqdm import tqdm
from collections import defaultdict

def parse_posts_from_markdown(file_content):
    posts = []
    post_blocks = file_content.strip().split('---\n\n')
    for block in post_blocks:
        lines = block.strip().split('\n')
        post = {
            'post_number': None,
            'reply_to_post_number': None,
            'username': '',
            'text': '',
            'url': '',
        }

        for line in lines:
            if line.startswith('## Post by '):
                name_match = re.search(r'## Post by (.*?) of username: (.*?) on', line)
                if name_match:
                    post['username'] = f"{name_match.group(1)} ({name_match.group(2)})"
            elif '**Post Number:**' in line:
                post['post_number'] = int(line.split('**Post Number:**')[-1].strip())
            elif '**Reply to Post Number:**' in line:
                reply_to = line.split('**Reply to Post Number:**')[-1].strip()
                post['reply_to_post_number'] = int(reply_to) if reply_to != 'None' else None
            elif '**Post URL:**' in line:
                post['url'] = line.split('**Post URL:**')[-1].strip()
            elif not line.startswith('**') and not line.startswith('##'):
                post['text'] += line + '\n'

        post['text'] = post['text'].strip()
        if post['post_number'] is not None:
            posts.append(post)

    return posts

def group_posts_as_qa(posts, topic_id):
    replies_map = defaultdict(list)

    for post in posts:
        if post['reply_to_post_number'] is not None:
            replies_map[post['reply_to_post_number']].append(post)

    grouped_qa = []
    for post in posts:
        if post['reply_to_post_number'] is None:  # root question
            answers = [{
                "text": reply['text'],
                "url": reply['url'],
                "answered_by": reply['username']
            } for reply in replies_map.get(post['post_number'], [])]

            grouped_qa.append({
                "question": post['text'],
                "question_url": post['url'],
                "asked_by": post['username'],
                "answers": answers,
                "topic_id": topic_id,
                "question_id": post['post_number'],
            })
    return grouped_qa

def process_all_markdown_files(folder_path):
    all_qa_pairs = []
    for md_file in tqdm(glob.glob(os.path.join(folder_path, '*.md')), desc="Processing markdown files"):
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        posts = parse_posts_from_markdown(content)
        topic_id = os.path.splitext(os.path.basename(md_file))[0]
        qa_pairs = group_posts_as_qa(posts, topic_id)
        all_qa_pairs.extend(qa_pairs)
    return all_qa_pairs

if __name__ == "__main__":
    discourse_md_folder = "discourse_md"
    qa_data = process_all_markdown_files(discourse_md_folder)

    with open("discourse_qa_pairs.jsonl", "w", encoding="utf-8") as f:
        for qa in qa_data:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    print(f"✅ Created {len(qa_data)} Q&A formatted entries with metadata and post URLs.")
