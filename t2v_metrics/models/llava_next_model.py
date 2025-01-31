import gc
import copy

import torch
from transformers import AutoTokenizer, LlavaNextForConditionalGeneration, LlavaNextProcessor
from t2v_metrics.visual_model import BaseVisualModel


LLAVA_NEXT_MODELS = {
    'llava-v1.6-7b-4bit': {
        'ckpt_path': 'unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit',
    },
}


class LLaVANextModel(BaseVisualModel):
    """A wrapper for the LLaVA-1.5 models"""

    def __init__(self, context_len: int = 2048):
        super(LLaVANextModel, self).__init__()
        self._question_template = 'Does this figure show "{}"? Please answer yes or no.'
        self._answer_template = "Yes"

        self._model: LlavaNextForConditionalGeneration = None
        self._processor: LlavaNextProcessor = None
        self._tokenizer = None

        self._context_len = context_len
        self._ignore_ind = -100

    def preload_model(self, model_name: str, torch_type: torch.dtype | None = None):
        """Load the model, tokenizer, image transform
        """
        self._model = LlavaNextForConditionalGeneration.from_pretrained(
            LLAVA_NEXT_MODELS[model_name]["ckpt_path"], torch_dtype=torch.float16, use_flash_attention_2=True, #load_in_4bit=True
        )
        self._model.to(self._device)
        self._processor = LlavaNextProcessor.from_pretrained(LLAVA_NEXT_MODELS[model_name]["ckpt_path"])
        self._tokenizer = AutoTokenizer.from_pretrained(LLAVA_NEXT_MODELS[model_name]["ckpt_path"])

    def unload_model(self):
        del self._model
        del self._processor
        del self._tokenizer

        torch.cuda.empty_cache()
        gc.collect()

    def format_answer(eslf, answer):
        answer = answer + "</s>"
        return answer

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
        content = [{"type": "text", "text": question}]
        for i in range(num_imgs):
            content.append({"type": "image"})


        messages = [{"role": "user",
                     "content": content}
                    # {
                    #     "role": "assistant",
                    #     "content": [
                    #         {"type": "text", "text": "Yes"}]
                    # }]
        ]
        return messages

    @torch.no_grad()
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def forward(
            self,
            images: list[torch.Tensor] | torch.Tensor,
            texts: list[str] | str,
            question_template: str = "",
            answer_template: str = ""
    ) -> torch.Tensor:
        """Forward pass of the model to return n scores for n (image, text) pairs (in PyTorch Tensor)
        """
        if question_template != "":
            self._question_template = question_template
        if answer_template != "":
            self._answer_template = answer_template

        questions = [self._question_template.format(text) for text in texts]
        answers = [self.format_answer(self._answer_template)] * len(texts)

        messages = self.create_message_template(len(images), questions[0])
        prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        print(prompt)

        inputs = self._processor(text=prompt,
                                 images=images,
                                 return_tensors="pt",
                                 padding=True,
                                 return_attention_mask=True
                                 )
        inputs = inputs.to(self._device)

        question_len = len(inputs['input_ids'][0])

        print(inputs)
        print(inputs["input_ids"].shape)

        # inputs consists of combined data: imagery + textual that were tokenized and preprocessed during call to processor
        tokens_to_append = self._tokenizer.encode(answers[0], return_tensors="pt")
        tokens_to_append = tokens_to_append[:, 1:].to(self._device)
        inputs["input_ids"] = torch.hstack([inputs["input_ids"], tokens_to_append])
        inputs["attention_mask"] = torch.hstack([inputs["attention_mask"], torch.ones_like(tokens_to_append)])

        print(inputs["input_ids"].shape)

        # setting image tokens to negative value, they will be ignored during inference#
        labels = copy.deepcopy(inputs["input_ids"]).to(self._device)
        labels[:, :question_len] = self._ignore_ind

        outputs = self._model(**inputs, labels=labels, return_dict=True)

        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        output_logits = outputs.logits[:, :-1, :].contiguous()
        output_labels = labels[:, 1:].contiguous()

        lm_prob = (-loss_fct(output_logits.view(-1, output_logits.size(-1)), output_labels.view(-1))).exp()
        return lm_prob