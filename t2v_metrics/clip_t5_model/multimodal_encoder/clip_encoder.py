import gc

import torch
import torch.nn as nn
from transformers import CLIPVisionModel


class CLIPVisionTower(nn.Module):
    def __init__(self, args, device: str = "cuda"):
        super().__init__()

        self._select_layer = args.mm_vision_select_layer
        self._select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self._device = device

        self.vision_tower: CLIPVisionModel = None

    def load_model(self, model_name: str):
        """"""
        self.vision_tower = CLIPVisionModel.from_pretrained(model_name)
        self.vision_tower.requires_grad_(False)
        # self._model.eval()

    def unload_model(self):
        """"""
        del self._model
        self._model = None

        gc.collect()
        torch.cuda.empty_cache()

    def feature_select(self, image_forward_outs: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        image_forward_outs

        Returns
        -------

        """
        image_features = image_forward_outs.hidden_states[self._select_layer]

        if self._select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self._select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self._select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images: torch.tensor):
        """

        Parameters
        ----------
        images

        Returns
        -------

        """
        image_forward_out = self.vision_tower(images.to(device=self._device), output_hidden_states=True)
        image_features = self.feature_select(image_forward_out).to(images.dtype)

        return image_features

    @property
    def hidden_size(self):
        """"""
        return self.vision_tower.config.hidden_size

    @property
    def num_patches(self):
        """"""
        return (self.vision_tower.config.image_size // self.vision_tower.config.patch_size) ** 2