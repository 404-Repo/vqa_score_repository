import copy
from typing import TypedDict

import torch
import torch.nn as nn

# from t2v_metrics.models.clip_t5_model.clip_t5_model import CLIPT5Model, CLIP_T5_MODELS
# from t2v_metrics.models.qwen2_vl_model import QwenVLModel, QWEN2_VL_MODELS
# from t2v_metrics.models.llava_model import LLaVAModel, LLAVA_MODELS
# from t2v_metrics.models.llava_next_model import LLaVANextModel, LLAVA_NEXT_MODELS
from t2v_metrics.models.deepseek_model import DeepSeekModel, DEEPSEEK_MODELS


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

    def preload_model(self, model_name: str, **kwargs):
        """

        Parameters
        ----------
        model_name

        Returns
        -------

        """
        # if model_name in CLIP_T5_MODELS.keys():
        #     self._model = CLIPT5Model()
        # elif model_name in LLAVA_MODELS.keys():
        #     self._model = LLaVAModel()
        # elif model_name in QWEN2_VL_MODELS.keys():
        #     self._model = QwenVLModel()
        # elif model_name in LLAVA_NEXT_MODELS.keys():
        #     self._model = LLaVANextModel()
        if model_name in DEEPSEEK_MODELS.keys():
            self._model = DeepSeekModel()
        else:
            ValueError(f"Passed model name <{model_name}> is not implemented or recognised.")

        assert self._model is not None
        self._model.preload_model(model_name, **kwargs)

    def unload_model(self):
        """"""
        self._model.unload_model()
