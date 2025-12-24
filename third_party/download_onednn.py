import os
import urllib.request
import ssl

def download_onednn():
    url = "https://github.com/oneapi-src/oneDNN/archive/refs/tags/v3.4.1.zip"
    os.makedirs("third_party", exist_ok=True)
    dest = os.path.join("third_party", "v3.4.1.zip")
    
    if os.path.exists(dest):
        print(f"File {dest} already exists. Skipping download.")
        return

    print(f"Downloading {url} to {dest}...")
    
    # Create an unverified context to bypass SSL errors
    context = ssl._create_unverified_context()
    
    try:
        with urllib.request.urlopen(url, context=context) as response, open(dest, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print("Download complete.")
    except Exception as e:
        print(f"Download failed: {e}")

if __name__ == "__main__":
    download_onednn()
