from hashlib import md5
from pathlib import Path
from zipfile import ZipFile

import requests
from tqdm import tqdm

# https://github.com/Junjue-Wang/LoveDA
download_url = "https://zenodo.org/api/records/5706578/files-archive"
fname = "LoveDA.zip"
hash = "37b8e126d6642151b3df5704cddded8d"


def download_large_file(url, save_path, chunk_size=1024 * 1024):
    """
    流式下载大文件到指定路径
    :param url: 待下载的URL
    :param save_path: 保存文件的完整路径
    :param chunk_size: 每次读取的字节数（默认1MB）
    """
    try:
        # stream=True 开启流式响应
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        # 分块写入文件
        with open(save_path, "wb") as f:
            for chunk in tqdm(response.iter_content(chunk_size=chunk_size)):
                if chunk:
                    f.write(chunk)
                    f.flush()

        print(f"成功下载到：{save_path}")
    except requests.exceptions.RequestException as e:
        print(f"下载失败：{e}")


def get_hash(path):
    with open(path, "rb") as f:
        data = f.read()
        return md5(data).hexdigest()


def check_file(path, hash):
    if not Path(fname).exists():
        return False
    if get_hash(fname) != hash:
        return False
    return True


def main():
    if hash is None:
        print("No hash provided!")
        exit(1)

    if not check_file(fname, hash):
        is_download = input("Hash check failed, download? (y/n)")
        if is_download.lower() == "y":
            download_large_file(download_url, fname)
        else:
            print("User canceled download.")
            exit(1)
    else:
        print("Hash check passed!")

    print("try to extract zip file")
    with ZipFile(fname, "r") as zip_file:
        zip_file.extractall("./dataset")

    print("try to extract dataset zip file")
    for file in Path("./dataset").glob("*.zip"):
        print(f"extracting {file.name}")
        with ZipFile(file, "r") as zip_file:
            zip_file.extractall(file.parent)


if __name__ == "__main__":
    main()
