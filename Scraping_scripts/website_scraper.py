import os
import json
import re
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from urllib.parse import urljoin, urlparse
from markdownify import markdownify as md
from playwright.sync_api import sync_playwright
from PIL import Image
from dotenv import load_dotenv
from google.generativeai import genai

BASE_URL = "https://tds.s-anand.net/#/2025-01/"
BASE_ORIGIN = "https://tds.s-anand.net"
OUTPUT_DIR = "tds_pages_md"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
METADATA_FILE = "metadata.json"

visited = set()
metadata = []

load_dotenv()
# Configure Gemini API

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

def sanitize_filename(title):
    return re.sub(r'[\\/*?:"<>|]', "_", title).strip().replace(" ", "_")

def extract_all_internal_links(page):
    links = page.eval_on_selector_all("a[href]", "els => els.map(el => el.href)")
    return list(set(
        link for link in links
        if BASE_ORIGIN in link and '/#/' in link
    ))

def wait_for_article_and_get_html(page):
    page.wait_for_selector("article.markdown-section#main", timeout=10000)
    return page.inner_html("article.markdown-section#main")

def describe_image_local(image_path):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        img = Image.open(image_path).convert("RGB")  # Convert to proper mode
        response = model.generate_content(["Describe the image in detail:", img])
        return response.text.strip()
    except Exception as e:
        print(f"❌ Failed to describe {image_path}: {e}")
        return "[Image description not available]"

def download_image(img_url):
    try:
        response = requests.get(img_url, timeout=10)
        response.raise_for_status()
        img_name = sanitize_filename(os.path.basename(urlparse(img_url).path))
        local_path = os.path.join("images", img_name)
        full_path = os.path.join(OUTPUT_DIR, local_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "wb") as f:
            f.write(response.content)
        return local_path  # relative path to be used in Markdown
    except Exception as e:
        print(f"⚠️ Failed to download image: {img_url}\n{e}")
        return None  # failed download

def fix_image_links(html, page_url):
    soup = BeautifulSoup(html, "html.parser")
    for img in soup.find_all("img"):
        src = img.get("src")
        if not src:
            continue

        abs_url = urljoin(page_url, src)
        local_path = download_image(abs_url)
        if not local_path:
            continue

        full_path = os.path.join(OUTPUT_DIR, local_path)
        description = describe_image_local(full_path)

        # Replace <img> with markdown image format
        markdown_img = soup.new_tag("p")
        markdown_img.string = f"![{description}]({local_path})"
        img.replace_with(markdown_img)

    return str(soup)

def crawl_page(page, url):
    if url in visited:
        return
    visited.add(url)

    print(f"📄 Visiting: {url}")
    try:
        page.goto(url, wait_until="domcontentloaded")
        page.wait_for_timeout(1000)
        raw_html = wait_for_article_and_get_html(page)
        html = fix_image_links(raw_html, url)
    except Exception as e:
        print(f"❌ Error loading page: {url}\n{e}")
        return

    title = page.title().split(" - ")[0].strip() or f"page_{len(visited)}"
    filename = sanitize_filename(title)
    filepath = os.path.join(OUTPUT_DIR, f"{filename}.md")

    markdown = md(html)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"---\n")
        f.write(f"title: \"{title}\"\n")
        f.write(f"original_url: \"{url}\"\n")
        f.write(f"downloaded_at: \"{datetime.now().isoformat()}\"\n")
        f.write(f"---\n\n")
        f.write(markdown)

    metadata.append({
        "title": title,
        "filename": f"{filename}.md",
        "original_url": url,
        "downloaded_at": datetime.now().isoformat()
    })

    links = extract_all_internal_links(page)
    for link in links:
        if link not in visited:
            crawl_page(page, link)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    global visited, metadata

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        crawl_page(page, BASE_URL)

        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✅ Completed. {len(metadata)} pages saved.")
        browser.close()

if __name__ == "__main__":
    main()




