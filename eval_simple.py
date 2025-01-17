import argparse
import os
import t2v_metrics
from pathlib import Path
import re


def get_all_image_files(folder_path: str) -> list[Path]:
    """"""
    folder = Path(folder_path)
    files_pngs = list(folder.rglob("*.png"))
    files_jpgs = list(folder.rglob("*.jpg"))
    files = files_pngs + files_jpgs

    sorted_files = sorted(files, key=lambda x: int(re.search(r'\d+', x.as_posix()).group()))

    if len(sorted_files) == 0:
        raise RuntimeWarning(f"No files were found in <{folder_path}>. Nothing to process!")

    return sorted_files


def main():
    img_folder = ""
    prompt = ""

    cache_dir = t2v_metrics.constants.HF_CACHE_DIR
    score_func = t2v_metrics.get_score_model(model="llava-v1.6-13b", device="cuda", cache_dir=cache_dir)

    imgs_paths = get_all_image_files(img_folder)
    scores = score_func.forward(imgs_paths, prompt)

    print("Scores: ", scores)
    print("Score: ", scores.mean())


if __name__ == '__main__':
    main()
