from typing import TypedDict

import torch
import torch.nn as nn

from .clip_t5_model.clip_t5_model import CLIPT5Model


class ImageTextDict(TypedDict):
    images: list[str]
    texts: list[str]


class VQAScore(nn.Module):
    def __init__(self, device="cuda:0"):
        """"""
        super().__init__()
        self._device = device
        self._model = CLIPT5Model(device)

    def forward(self, images: list[torch.Tensor] | torch.Tensor, texts: list[str] | str, **kwargs):
        """

        Parameters
        ----------
        images
        texts
        kwargs

        Returns
        -------

        """

        if isinstance(images, torch.Tensor):
            images = [images]
        if isinstance(texts, str):
            texts = [texts]

        # scores = torch.zeros(len(images), len(texts)).to(self._device)
        # for i, image in enumerate(images):
        scores = self._model.forward(images * len(texts), texts, **kwargs)
        return scores

    def preload_model(self, model_name: str):
        """"""
        self._model.preload_model(model_name)

    def unload_model(self):
        """"""
        self._model.unload_model()


