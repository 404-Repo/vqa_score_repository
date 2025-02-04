import gc
import copy

import torch
from transformers import AutoTokenizer, LlavaForConditionalGeneration, LlavaProcessor
from t2v_metrics.visual_model import BaseVisualModel


LLAVA_MODELS = {
    'llava-v1.5-13b': {
        'tokenizer' : {
            'path': 'llava-hf/llava-1.5-13b-hf',
        },
        'model': {
            'path': 'llava-hf/llava-1.5-13b-hf',
            'conversation': 'chat',
            'image_aspect_ratio': 'pad',
        },
    },
    'llava-v1.5-7b': {
        'tokenizer' : {
            'path': 'llava-hf/llava-1.5-7b-hf',
        },
        'model': {
            'path': 'llava-hf/llava-1.5-7b-hf',
            'conversation': 'chat',
            'image_aspect_ratio': 'pad',
        },
    },
    'llava-v1.5-7b-int4': {
        'tokenizer': {
            'path' : 'unsloth/llava-1.5-7b-hf-bnb-4bit'
        },
        'model': {
            'path': 'unsloth/llava-1.5-7b-hf-bnb-4bit',
            'conversation': 'chat',
            'image_aspect_ratio': 'pad'
        }
    },
}


class LLaVAModel(BaseVisualModel):
    """A wrapper for the LLaVA-1.5 models"""
    def __init__(self, context_len: int = 2048):
        super(LLaVAModel, self).__init__()
        self._question_template = 'Does this figure show "{}"? Please answer yes or no.'
        self._answer_template = "Yes"

        self._model: LlavaForConditionalGeneration = None
        self._processor: LlavaProcessor = None
        self._tokenizer = None

        self._context_len = context_len
        self._ignore_ind = -100

    def preload_model(self, model_name: str, quant_type: dict = {}):
        """Load the model, tokenizer, image transform
        """
        self._model = LlavaForConditionalGeneration.from_pretrained(LLAVA_MODELS[model_name]["model"]["path"], torch_dtype=torch.bfloat16)
        self._model.to(self._device)
        self._processor = LlavaProcessor.from_pretrained(LLAVA_MODELS[model_name]["model"]["path"])
        self._tokenizer = AutoTokenizer.from_pretrained(LLAVA_MODELS[model_name]["tokenizer"]["path"])

    def unload_model(self):
        del self._model
        del self._processor
        del self._tokenizer

        torch.cuda.empty_cache()
        gc.collect()

    def format_question(self, question: str, conversation_style: str='chat'):
        system_message = ("A chat between a curious user and an artificial intelligence assistant. "
                          "The assistant gives helpful, detailed, and polite answers to the user's questions.")

        if conversation_style == 'plain':  # for 1st stage model
            question = "<image>" + question
        elif conversation_style == 'chat':  # for 2nd stage model
            question = system_message + " USER: " + "<image>" + "\n" + question + " ASSISTANT: "
        else:
            raise NotImplementedError()
        return question

    def format_answer(eslf, answer, conversation_style='chat'):
        if conversation_style == 'plain':  # for 1st stage model
            answer = answer + "\n"
        elif conversation_style == 'chat':  # for 2nd stage model
            answer = answer + "</s>"
        else:
            raise NotImplementedError()
        return answer

    @torch.no_grad()
    @torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    def forward(
            self,
            images: list[torch.Tensor] | torch.Tensor,
            texts: list[str] | str,
            question_template: str = "",
            answer_template: str = ""
    ) -> torch.Tensor:
        """Forward pass of the model to return n scores for n (image, text) pairs (in PyTorch Tensor)
        """
        assert len(images) == len(texts), "Number of images and texts must match"
        # Turn "a photo of a dog" into
        # Q: "Does this figure show "a photo of a dog"? Please answer yes or no."
        # A: "Yes"
        questions = [self._question_template.format(text) for text in texts]
        answers = [self._answer_template.format(text) for text in texts]
        
        # Formatting for LLaVA-1.5 desired input including system message and image tokens
        questions = [self.format_question(question, conversation_style="chat") for question in questions]
        answers = [self.format_answer(answer, conversation_style="chat") for answer in answers]

        inputs = self._processor(images=images, text=questions, tokenizer=self._tokenizer, return_tensors="pt", return_attention_mask=True)
        question_len = len(inputs['input_ids'][0])

        tokens_to_append = torch.tensor(self._tokenizer.encode(answers[0]))[1:]

        inputs["input_ids"] = torch.hstack([inputs["input_ids"].squeeze(0), tokens_to_append]).unsqueeze(0)
        inputs["attention_mask"] = torch.hstack([inputs["attention_mask"].squeeze(0), torch.tensor([1]*len(tokens_to_append))]).unsqueeze(0)

        labels = copy.deepcopy(inputs["input_ids"])
        labels[:, :question_len] = self._ignore_ind
        
        # assert input_ids is None, "input_ids should be None for LLaVA-1.5"
        # assert past_key_values is None, "past_key_values should be None for LLaVA-1.5"
        model_input_kwargs = {
            'input_ids': inputs["input_ids"].to(self._device),
            'attention_mask': inputs["attention_mask"].to(self._device),
            'pixel_values': inputs["pixel_values"].to(self._device),
            'labels': labels.to(self._device),
            'inputs_embeds': None,
            'use_cache': None,
            'output_attentions': None,
            'output_hidden_states': None,
            'return_dict': True,
        }
        
        outputs = self._model(
            **model_input_kwargs
        )

        logits = outputs["logits"]

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        shift_labels = shift_labels.to(shift_logits.device)
        lm_prob = torch.zeros(shift_logits.shape[0])
        for k in range(lm_prob.shape[0]):
            lm_prob[k] = (-loss_fct(shift_logits[k], shift_labels[k])).exp()

        return lm_prob