from time import time
from pathlib import Path

import torch
import numpy as np
import t2v_metrics
from PIL import Image


if __name__ == '__main__':
    image_path = "./images/image.png"
    prompt1 = "two dogs running in the forest"

    image = np.array(Image.open(image_path))
    torch_image = torch.tensor(image)

    model = t2v_metrics.VQAScore()
    model.preload_model("clip-flant5-xl")

    t1 = time()
    score1 = model([torch_image], [prompt1])
    t2 = time()

    print(f" It took: {t2 - t1} s")
    print(f" Input prompt: {prompt1}")
    print(f" VQAScore: {score1}")

    image_folder = Path("./images/model_renders")
    prompt2 = "oak tree low poly"

    images_files = list(image_folder.rglob("*.png"))
    images = []
    for img_file in images_files:
        image = torch.tensor(np.array(Image.open(img_file)))
        images.append(image)

    t1 = time()
    score2 = model(images, [prompt2])
    t2 = time()

    print(f" It took: {t2 - t1} s")
    print(f" Input prompt: {prompt2}")
    print(f" VQAScore: {score2}")