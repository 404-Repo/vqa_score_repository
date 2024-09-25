import gc

import torch
import torch.nn as nn
import open_clip
from open_clip import CLIP
from open_clip.tokenizer import HFTokenizer


class CLIPVisionTower(nn.Module):
    def __init__(self, args, device: str = "cuda"):
        super().__init__()

        self._select_layer = args.mm_vision_select_layer
        self._select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self._device = device

        self.vision_tower: CLIP = None
        self._tokenizer: HFTokenizer = None

    def load_model(self, model_name: str = "", pretrained: str = ""):
        """

        Parameters
        ----------
        model_name
        pretrained

        Returns
        -------

        """
        self.vision_tower, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self._device, output_dict=True
        )
        self._tokenizer = open_clip.get_tokenizer(model_name)
        self.vision_tower.requires_grad_(False)

    def unload_model(self):
        """"""
        del self.vision_tower
        del self._tokenizer
        self.vision_tower = None
        self._tokenizer = None

        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def forward(self, images: torch.Tensor):
        """

        Parameters
        ----------
        images

        Returns
        -------

        """

        global hidden_states
        hidden_states = []

        def get_hidden_states(module, input, output):
            hidden_states.append(output)

        layer_to_hook = self.vision_tower.visual.transformer.resblocks[self._select_layer]
        hook = layer_to_hook.register_forward_hook(get_hidden_states)

        with torch.no_grad():
            self.vision_tower.encode_image(images)

            hidden_states = torch.stack(hidden_states).to(images.dtype).squeeze(0)

            if self._select_feature == 'patch':
                images_features = hidden_states[:, 1:]
            elif self._select_feature == 'cls_patch':
                images_features = hidden_states
            else:
                raise ValueError(f'Unexpected select feature: {self._select_feature}')
        hook.remove()

        return images_features

    @property
    def hidden_size(self):
        """"""
        return self.vision_tower.config.hidden_size

    @property
    def num_patches(self):
        """"""
        return (self.vision_tower.config.image_size // self.vision_tower.config.patch_size) ** 2
