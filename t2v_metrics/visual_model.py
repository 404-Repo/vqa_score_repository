from abc import ABC, abstractmethod

import torch


class BaseVisualModel(ABC):

    @abstractmethod
    @torch.no_grad()
    @torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    def forward(self, images: list[torch.Tensor] | torch.Tensor,
                texts: list[str] | str,
                question_template: str,
                answer_template: str) -> torch.Tensor:
        """

        Parameters
        ----------
        images
        texts
        question_template
        answer_template

        Returns
        -------

        """
        pass

    @abstractmethod
    def preload_model(self, model_name: str):
        """

        Parameters
        ----------
        model_name

        Returns
        -------

        """
        pass

    @abstractmethod
    def unload_model(self):
        """

        Returns
        -------

        """
        pass

