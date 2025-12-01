import os
import json
import requests
from pathlib import Path
from tqdm import tqdm

class LidarDownloader:
    def __init__(self, config):
        self.config = config

    def download_file(self, url, local_filename):
        local_path = os.path.join(self.config['CACHE_DIR'], local_filename)
        if os.path.exists(local_path) and os.path.getsize(local_path) > 10240: return local_path
        
        print(f"   ⬇️ Downloading: {local_filename}...")
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                if r.status_code != 200: 
                    print(f"   ❌ HTTP Error {r.status_code} for {url}")
                    return None
                temp_path = local_path + ".tmp"
                with open(temp_path, 'wb') as f, tqdm(
                    desc=local_filename,
                    total=int(r.headers.get('content-length', 0)),
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for chunk in r.iter_content(8192): 
                        size = f.write(chunk)
                        bar.update(size)
                if os.path.getsize(temp_path) < 1024: 
                    os.remove(temp_path)
                    return None
                os.rename(temp_path, local_path)
                return local_path
        except Exception as e: 
            print(f"   ❌ Error: {e}")
            return None

    def run(self):
        # Ensure cache directory exists
        Path(self.config['CACHE_DIR']).mkdir(parents=True, exist_ok=True)
        
        print("1. Loading Selected Tiles List...")
        if not os.path.exists(self.config['OUTPUT_TILES_PATH']):
            print(f"❌ Selected tiles list not found: {self.config['OUTPUT_TILES_PATH']}")
            return

        with open(self.config['OUTPUT_TILES_PATH']) as f:
            selected_tiles = json.load(f)
        
        print(f"   Found {len(selected_tiles)} tiles to download.")

        print("2. Downloading Tiles...")
        for tile in selected_tiles:
            url = tile['url']
            filename = tile['filename']
            self.download_file(url, filename)
        
        print("✅ Download complete.")
