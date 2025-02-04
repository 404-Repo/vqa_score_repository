import copy
import gc

import torch
from transformers import (PaliGemmaPreTrainedModel, PaliGemmaProcessor, AutoTokenizer)
from t2v_metrics.visual_model import BaseVisualModel


PALIGEMMA2_MODELS = {
    'paligemma2-3b-448': {
        'ckpt_path': 'google/paligemma2-3b-pt-448'
    },
}


class PaligemmaModel(BaseVisualModel):
    """"""
    def __init__(self, context_len:int=2048):
        """

        Parameters
        ----------
        device
        """
        super(PaligemmaModel, self).__init__()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._question_template = 'Does this figure show "{}"? Please answer yes or no.'
        self._answer_template = "Yes"
        self._image_token = ""

        self._processor = None
        self._model = None
        self._tokenizer = None
        self._context_len = context_len
        self._padding = -100

    def preload_model(self, model_name: str, quant_type: dict = {}):
        """

        Parameters
        ----------
        model_name
        torch_type

        Returns
        -------

        """
        self._model = PaliGemmaPreTrainedModel.from_pretrained(
            PALIGEMMA2_MODELS[model_name]["ckpt_path"],
            torch_dtype=torch.bfloat16,
            load_in_8bit=True,
            trust_remote_code=True,
            device_map="auto"
        )
        self._model.eval()

        self._processor = PaliGemmaProcessor.from_pretrained(PALIGEMMA2_MODELS[model_name]["ckpt_path"], trust_remote_code=True)
        self._tokenizer = AutoTokenizer.from_pretrained(
            PALIGEMMA2_MODELS[model_name]["ckpt_path"], trust_remote_code=True, use_fast=False
        )

    def unload_model(self):
        """"""
        del self._model
        del self._processor
        del self._tokenizer
        torch.cuda.empty_cache()
        gc.collect()

        self._model = None
        self._processor = None
        self._tokenizer = None

    def format_answer(self, answer: str):
        """

        Parameters
        ----------
        answer

        Returns
        -------

        """
        return answer

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
        if question_template != "":
            self._question_template = question_template
        if answer_template != "":
            self._answer_template = answer_template

        questions = [self._question_template.format(text) for text in texts]
        answers = [self.format_answer(self._answer_template)] * len(texts)

        inputs = self._processor(text=questions[0], images = images, return_tensors="pt")

        print(inputs.keys())

        question_len = len(inputs['input_ids'])

        # inputs consists of combined data: imagery + textual that were tokenized and preprocessed during call to processor
        tokens_to_append = self._tokenizer.encode(answers[0], return_tensors="pt")
        tokens_to_append = tokens_to_append[:, 1:]
        inputs["input_ids"] = torch.hstack([inputs["input_ids"], tokens_to_append]).to(self._device)
        inputs["attention_mask"] = torch.hstack([inputs["attention_mask"], torch.ones_like(tokens_to_append)]).to(self._device)

        # setting image tokens to negative value, they will be ignored during inference#
        labels = copy.deepcopy(inputs["input_ids"]).to(self._device)
        labels[:, :question_len] = self._padding

        outputs = self._model(**inputs, labels=labels, return_dict=True)

        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        output_logits = outputs.logits[:, :-1, :].contiguous()
        output_labels = labels[:, 1:].contiguous()

        lm_prob = (-loss_fct(output_logits.view(-1, output_logits.size(-1)), output_labels.view(-1))).exp()
        return lm_prob
