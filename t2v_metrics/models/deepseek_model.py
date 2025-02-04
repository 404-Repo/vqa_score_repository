import gc
import copy
from PIL import Image

import torch
from t2v_metrics.models.deepseek_vl2_model import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from t2v_metrics.visual_model import BaseVisualModel


DEEPSEEK_MODELS = {
    'deepseek-vl2-tiny': {
        'ckpt_path': 'deepseek-ai/deepseek-vl2-tiny',
    },
}


class DeepSeekModel(BaseVisualModel):
    """A wrapper for the LLaVA-1.5 models"""

    def __init__(self, context_len: int = 2048):
        super(DeepSeekModel, self).__init__()
        self._question_template = 'Does this figure show "{}"? Please answer yes or no.'
        self._answer_template = "Yes"

        self._model = None
        self._processor = None
        self._tokenizer = None

        self._context_len = context_len
        self._ignore_ind = -100

    def preload_model(self, model_name: str, quant_type: dict = {}):
        """Load the model, tokenizer, image transform
        """
        self._model = DeepseekVLV2ForCausalLM.from_pretrained(
            DEEPSEEK_MODELS[model_name]["ckpt_path"],
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto"
        )
        self._model.eval()

        self._processor = DeepseekVLV2Processor.from_pretrained(DEEPSEEK_MODELS[model_name]["ckpt_path"])
        self._tokenizer = self._processor.tokenizer

    def unload_model(self):
        del self._model
        del self._processor
        del self._tokenizer

        torch.cuda.empty_cache()
        gc.collect()

    def format_answer(self, answer):
        answer = answer
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
        content = ""
        for i in range(1, num_imgs+1):
            content += f"image_{i}:<image>\n"
        content += question

        messages = [
            {
                "role": "<|User|>",
                "content": content,
            },
            {
                "role": "<|Assistant|>",
                "content": ""
            }
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

        images = [Image.fromarray(img.detach().cpu().numpy()) for img in images]

        messages = self.create_message_template(len(images), questions[0])
        inputs = self._processor(conversations=messages, images=images, force_batchify=True, inference_mode=True, system_prompt="")
        question_len = len(inputs.input_ids[0])

        # inputs consists of combined data: imagery + textual that were tokenized and preprocessed during call to processor
        tokens_to_append = self._tokenizer.encode(answers[0], return_tensors="pt", add_special_tokens = False)
        tokens_to_append = tokens_to_append[:, :]

        inputs.input_ids = torch.hstack([inputs.input_ids, tokens_to_append])
        inputs.attention_mask = torch.hstack([inputs.attention_mask, torch.ones_like(tokens_to_append)])
        inputs.images_seq_mask = torch.hstack([inputs.images_seq_mask, torch.zeros_like(tokens_to_append).to(torch.bool)])

        # setting image tokens to negative value, they will be ignored during inference#
        labels = copy.deepcopy(inputs.input_ids)
        labels[:, :question_len] = self._ignore_ind
        inputs.labels = labels

        input_args = {
            "input_ids": inputs.input_ids.to(self._device),
            "attention_mask": inputs.attention_mask.to(self._device),
            "images": inputs.images.to(self._device),
            "images_seq_mask": inputs.images_seq_mask.to(self._device),
            "images_spatial_crop": inputs.images_spatial_crop.to(self._device),
            "labels": inputs.labels.to(self._device),
            "use_cache": False
        }

        outputs = self._model(**input_args, return_dict=True)

        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        output_logits = outputs.logits[:, :-1, :].contiguous()
        output_labels = labels[:, 1:].contiguous()

        lm_prob = (-loss_fct(output_logits.view(-1, output_logits.size(-1)), output_labels.view(-1))).exp()
        return lm_prob
