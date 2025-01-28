import copy
from typing import TypedDict

import torch
import torch.nn as nn

from t2v_metrics.models.clip_t5_model.clip_t5_model import CLIPT5Model
from t2v_metrics.models.smolvlm_model import SmolVLMModel
from t2v_metrics.models.llava_model import LLaVAModel
from t2v_metrics.models.cogvlm_model import CogVLMModel


class ImageTextDict(TypedDict):
    images: list[str]
    texts: list[str]


class VQAScore(nn.Module):
    def __init__(self):
        """"""
        super().__init__()
        self._model = None

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
        if model_name == "smolvlm-1.7b-base" or model_name == "smolvlm-1.7b-synth":
            self._model = SmolVLMModel()
            self._model.preload_model(model_name)
        elif model_name == "clip-flant5-xxl" or model_name == "clip-flant5-xl":
            self._model = CLIPT5Model()
            self._model.preload_model(model_name)
        elif model_name == "llava-v1.5-7b" or model_name == "llava-v1.5-13b":
            self._model = LLaVAModel()
            self._model.preload_model(model_name)
        elif model_name == "cogvlm-17b":
            self._model = CogVLMModel()
            self._model.preload_model(model_name)


    def unload_model(self):
        """"""
        self._model.unload_model()
