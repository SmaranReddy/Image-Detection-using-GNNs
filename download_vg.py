import os
import urllib.request
import zipfile

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "visual_genome")
os.makedirs(BASE_DIR, exist_ok=True)

FILES = {
    "relationships.json": "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/relationships.json.zip",
    "image_data.json":    "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip",
}

for filename, url in FILES.items():
    dest = os.path.join(BASE_DIR, filename)
    if os.path.exists(dest):
        print(f"[skip] {filename} already exists.")
        continue

    zip_path = dest + ".zip"
    print(f"[download] {filename} ...")
    urllib.request.urlretrieve(url, zip_path)

    print(f"[extract] {filename} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(BASE_DIR)
    os.remove(zip_path)
    print(f"[done] {filename}")

print("\n--- Verification ---")
for filename in FILES:
    path = os.path.join(BASE_DIR, filename)
    status = "OK" if os.path.exists(path) else "MISSING"
    print(f"[{status}] {path}")
