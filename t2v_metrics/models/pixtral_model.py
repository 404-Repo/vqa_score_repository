import copy
import gc

import torch
from transformers import (AutoTokenizer,
                          AutoProcessor,
                          LlavaForConditionalGeneration)
from t2v_metrics.visual_model import BaseVisualModel


PIXTRAL_MODELS = {
    'pixtral-12b-8bit': {
        'ckpt_path': 'DewEfresh/pixtral-12b-8bit',
    },
}


class PixtralVisualModel(BaseVisualModel):
    """"""
    def __init__(self, context_len:int=2048):
        """

        Parameters
        ----------
        device
        """
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._question_template = 'Does this figure show "{}"? Please answer yes or no.'
        self._answer_template = "Yes"
        self._image_token = ""

        self._processor = None
        self._model = None
        self._tokenizer = None
        self._context_len = context_len
        self._padding = -100

    def preload_model(self, model_name: str, torch_type: torch.dtype | None = None):
        """

        Parameters
        ----------
        model_name
        torch_type

        Returns
        -------

        """
        self._model = LlavaForConditionalGeneration.from_pretrained(
            PIXTRAL_MODELS[model_name]["ckpt_path"],
            device_map="auto"
        )
        self._model.eval()

        self._processor = AutoProcessor.from_pretrained(
            PIXTRAL_MODELS[model_name]["ckpt_path"]
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            PIXTRAL_MODELS[model_name]["ckpt_path"]
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
        return "\n" + answer + "<|end|>"

    def create_message_template(self, num_imgs: int, question: str):
        """

        Parameters
        ----------
        num_imgs
        question
        answer

        Returns
        -------

        """
        image_placeholder = ""
        for i in range(num_imgs):
            image_placeholder += f"<|image_{i}|>\n"

        messages = [{"role": "user",
                     "content": [{"type": "image"}] * num_imgs + [{"type": "text", "content": question}]},
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "content": "Yes"}]
                    }
                    ]
        return messages

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

        messages = self.create_message_template(len(images), questions[0])
        prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        print(prompt)

        # images = [Image.fromarray(img.detach().cpu().numpy()) for img in images]
        #
        # inputs = self._processor(text=prompt,
        #                          images = images,
        #                          return_tensors="pt",
        #                          )
        # inputs = inputs.to(self._device)#
        #
        # question_len = len(inputs['input_ids'][0])
        #
        # # inputs consists of combined data: imagery + textual that were tokenized and preprocessed during call to processor
        # tokens_to_append = self._tokenizer.encode(answers[0], return_tensors="pt")
        # tokens_to_append = tokens_to_append[:, 1:].to(self._device)
        # inputs["input_ids"] = torch.hstack([inputs["input_ids"], tokens_to_append])
        # inputs["attention_mask"] = torch.hstack([inputs["attention_mask"], torch.ones_like(tokens_to_append)])
        #
        # # setting image tokens to negative value, they will be ignored during inference#
        # labels = copy.deepcopy(inputs["input_ids"]).to(self._device)
        # labels[:, :question_len] = self._padding
        #
        # outputs = self._model(**inputs, labels=labels, return_dict=True)
        #
        # loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        # output_logits = outputs.logits[:, :-1, :].contiguous()
        # output_labels = labels[:, 1:].contiguous()
        #
        # lm_prob = (-loss_fct(output_logits.view(-1, output_logits.size(-1)), output_labels.view(-1))).exp()
        return 0 #lm_prob
