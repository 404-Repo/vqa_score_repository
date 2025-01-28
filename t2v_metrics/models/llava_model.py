import gc
import copy

import torch
from transformers import AutoTokenizer, LlavaForConditionalGeneration, LlavaProcessor
from t2v_metrics.visual_model import BaseVisualModel


LLAVA_MODELS = {
    'llava-v1.5-13b': {
        'tokenizer' : {
            'path': 'liuhaotian/llava-v1.5-13b',
        },
        'model': {
            'path': 'liuhaotian/llava-v1.5-13b',
            'conversation': 'chat',
            'image_aspect_ratio': 'pad',
        },
    },
    'llava-v1.5-7b': {
        'tokenizer' : {
            'path': 'liuhaotian/llava-v1.5-7b',
        },
        'model': {
            'path': 'liuhaotian/llava-v1.5-7b',
            'conversation': 'chat',
            'image_aspect_ratio': 'pad',
        },
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
        self._padding = -100
        self._ignore_ind = -200

    def preload_model(self, model_name: str, torch_type: torch.dtype | None = None):
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

    def tokenizer_image_token(self, prompt, tokenizer, image_token_index=-200, return_tensors=None):
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids

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
        questions = [question_template.format(text) for text in texts]
        answers = [answer_template.format(text) for text in texts]
        
        # Formatting for LLaVA-1.5 desired input including system message and image tokens
        questions = [self.format_question(question, conversation_style="chat") for question in questions]
        answers = [self.format_answer(answer, conversation_style="chat") for answer in answers]
        
        # images = self.load_images(images)
        prompts = [qs + ans for qs, ans in zip(questions, answers)]
        inputs = self._processor(images=images, text=prompts, return_tensors="pt", return_attention_mask=True)

        input_ids = inputs["input_ids"]
        labels = copy.deepcopy(input_ids)
        for label, qs in zip(labels, questions):
            tokenized_len = len(self.tokenizer_image_token(qs, self._tokenizer))
            if qs[-1] == " ":
                tokenized_len -= 1 # because white space
            label[:tokenized_len] = self._ignore_ind
    
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self._tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                batch_first=True,
                                                padding_value=self._padding)
        input_ids = input_ids[:, :self._tokenizer.model_max_length]
        labels = labels[:, :self._tokenizer.model_max_length]
            
        # attention_mask = input_ids.ne(self._tokenizer.pad_token_id)
        # input_ids, attention_mask, labels = input_ids.to(self._device), attention_mask.to(self._device), labels.to(self._device)
        # input_ids, attention_mask, past_key_values, inputs_embeds, labels = self._model.prepare_inputs_labels_for_multimodal(
        #     input_ids,
        #     attention_mask,
        #     None,
        #     labels,
        #     images
        # )
        
        # assert input_ids is None, "input_ids should be None for LLaVA-1.5"
        # assert past_key_values is None, "past_key_values should be None for LLaVA-1.5"
        model_input_kwargs = {
            'input_ids': input_ids,
            'attention_mask': inputs["attention_mask"].to(self._device),
            'pixel_values': inputs["pixel_values"].to(self._device),
            'inputs_embeds': None,
            'use_cache': None,
            'output_attentions': None,
            'output_hidden_states': None,
            'return_dict': False,
        }
        
        outputs = self._model(
            **model_input_kwargs
        )

        hidden_states = outputs[0]
        logits = self._model.lm_head(hidden_states)

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