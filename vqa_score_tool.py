from time import time
from pathlib import Path

import torch
import numpy as np
import t2v_metrics
from PIL import Image


if __name__ == '__main__':
    image_path = "./images/image.png"
    # image_path = "./images/circle.jpg"
    prompt1 = "the brown dog chases the black dog around the tree"
    # prompt1 = "a perfect circle with black border"

    image = np.array(Image.open(image_path))
    torch_image = torch.tensor(image)

    model = t2v_metrics.VQAScore()
    model.preload_model("llava-v1.6-vicuna-7b")
    # model.preload_model("qwen2-vl-2b")
    # model.preload_model("qwen2-vl-7b-int8")
    # model.preload_model("llava-v1.5-7b")

    t1 = time()
    score1 = model([torch_image], [prompt1])
    t2 = time()

    print(f" It took: {t2 - t1} s")
    print(f" Input prompt: {prompt1}")
    print(f" VQAScore: {score1}")

    # # image_folder = Path("/home/tesha/Documents/Python/three-gen-subnet-private/validation/benchmark/benchmark_output/low_quality/images/happy_imp_")
    # image_folder = Path("/home/tesha/Documents/Python/three-gen-subnet-private/validation/benchmark/benchmark_output/medium_quality/images/happy_imp")
    # # prompt2 = "oak tree low poly"
    # prompt2 = "happy imp"
    #
    # images_files = list(image_folder.rglob("*.png"))
    # images = []
    # for img_file in images_files:
    #     image = torch.tensor(np.array(Image.open(img_file)))
    #     images.append(image)
    #
    # t1 = time()
    # score2 = model(images, [prompt2])
    # t2 = time()
    #
    # score2 = np.exp(np.log(score2).mean())
    #
    # print(f" It took: {t2 - t1} s")
    # print(f" Input prompt: {prompt2}")
    # print(f" VQAScore: {score2}")