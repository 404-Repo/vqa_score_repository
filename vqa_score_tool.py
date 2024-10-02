from time import time
from pathlib import Path

import numpy as np
import t2v_metrics
from PIL import Image


if __name__ == '__main__':
    image_path = "./images/image.png"
    prompt1 = "two dogs running in the forest"

    image = Image.open(image_path)
    model = t2v_metrics.VQAScore(model="clip-flant5-xl")

    t1 = time()
    score1 = model([image], [prompt1])
    t2 = time()

    print(f" It took: {t2 - t1} s")
    print(f" Input prompt: {prompt1}")
    print(f" VQAScore: {score1}")

    image_folder = Path("./images/model_renders")
    prompt2 = "oak tree low poly"

    images_files = list(image_folder.rglob("*.png"))
    images = []
    for img_file in images_files:
        image = Image.open(img_file)
        images.append(image)

    t1 = time()
    score2 = model(images, [prompt2])
    t2 = time()

    score2 = np.exp(np.log(score2.detach().cpu().numpy()).mean())

    print(f" It took: {t2 - t1} s")
    print(f" Input prompt: {prompt2}")
    print(f" VQAScore: {score2}")
