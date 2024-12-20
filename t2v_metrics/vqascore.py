import copy
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

    def forward(self, images: list[torch.Tensor], texts: list[str], **kwargs):
        """

        Parameters
        ----------
        images
        texts
        kwargs

        Returns
        -------

        """

        if len(images) > len(texts):
            texts = [copy.deepcopy(texts[0]) for i in range(len(images))]
        else:
            images = [copy.deepcopy(images[0]) for i in range(len(texts))]

        scores = self._model.forward(images, texts, **kwargs)
        return scores

    def preload_model(self, model_name: str):
        """

        Parameters
        ----------
        model_name

        Returns
        -------

        """
        self._model.preload_model(model_name)

    def unload_model(self):
        """"""
        self._model.unload_model()


