from time import time

import torch
import numpy as np
import t2v_metrics
from PIL import Image


if __name__ == '__main__':
    image_path = "./images/image.png"
    prompt = "two dogs running in the forest"

    image = np.array(Image.open(image_path))
    torch_image = torch.tensor(image)

    model = t2v_metrics.VQAScore()
    model.preload_model("clip-flant5-xl")

    t1 = time()
    score = model(torch_image, prompt)
    t2 = time()

    print(f" It took: {t2 - t1} s")

    print(f" Input prompt: {prompt}")
    print(f" VQAScore: {score}")
