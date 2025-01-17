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


def load_prompts(folder_path: Path) -> list[str]:
    """
    Function for loading prompts from the specified txt file

    Parameters
    ----------
    folder_path: path to the file with prompts

    Returns
    -------
    prompts_in: a list with preloaded prompts
    """
    prompt_file = list(folder_path.rglob("*.txt"))
    if len(prompt_file) == 0:
        prompt_file = list(folder_path.parent.rglob("*.txt"))
    else:
        print(f"Cannot find txt file with prompts in folder: {folder_path} or {folder_path.parent}")

    with prompt_file[0].open() as file:
        prompts_in = [line.rstrip() for line in file]

    return prompts_in


def main():
    img_folder = "./test_data"

    prompts = load_prompts(img_folder)

    cache_dir = t2v_metrics.constants.HF_CACHE_DIR
    score_func = t2v_metrics.get_score_model(model="llava-v1.6-13b", device="cuda", cache_dir=cache_dir)

    imgs_paths = get_all_image_files(img_folder)
    for img_file, prompt in zip(imgs_paths, prompts):
        score = score_func.forward([imgs_file], [prompt])
        print("Scores: ", score)


if __name__ == '__main__':
    main()
