from abc import ABC, abstractmethod

import torch


class BaseVisualModel(ABC):
    @abstractmethod
    def __init__(self):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self._device)

    @abstractmethod
    @torch.no_grad()
    @torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    def forward(self, images: list[torch.Tensor] | torch.Tensor,
                texts: list[str] | str,
                question_template: str = "",
                answer_template: str = "") -> torch.Tensor:
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
    def preload_model(self, model_name: str, **kwargs):
        """

        Parameters
        ----------
        model_name
        torch_type

        Returns
        -------

        """
        pass

    @abstractmethod
    def unload_model(self):
        """"""
        pass

    @abstractmethod
    def format_answer(self, answer: str):
        """

        Parameters
        ----------
        answer

        Returns
        -------

        """
        pass

