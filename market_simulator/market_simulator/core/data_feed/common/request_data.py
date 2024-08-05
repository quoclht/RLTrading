from urllib import request
from urllib.error import HTTPError

def download_file(file_url: str, file_path: str):
    print(f"Downloading {file_url}")
    try:
        dl_file = request.urlopen(file_url)
        length = dl_file.getheader("content-length")
        if length:
            length = int(length)
            blocksize = max(4096, length // 100)

        with open(file_path, "wb") as out_file:
            dl_progress = 0
            while True:
                buf = dl_file.read(blocksize)
                if not buf:
                    break
                dl_progress += len(buf)
                out_file.write(buf)

    except HTTPError:
        print("\nFile not found: {}".format(file_url))
        pass