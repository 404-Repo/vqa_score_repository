import gc

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from torchvision import transforms

from t2v_metrics.visual_model import BaseVisualModel
from .language_model.clip_t5 import CLIPT5ForConditionalGeneration


CLIP_T5_MODELS = {
    # We recommend using 'clip-flant5-xxl' for maximal performance.
    # If you want to use a smaller model, we recommend using 'clip-flant5-xl'.
    'clip-flant5-xxl': {
        'tokenizer': {
            'path': 'google/flan-t5-xxl',
            'model_max_length': 2048,
        },
        'model': {
            'path': 'zhiqiulin/clip-flant5-xxl',
            'conversation': 't5_chat',
            'image_aspect_ratio': 'pad',
        },
    },
    'clip-flant5-xl': {
        'tokenizer': {
            'path': 'google/flan-t5-xl',
            'model_max_length': 2048,
        },
        'model': {
            'path': 'zhiqiulin/clip-flant5-xl',
            'conversation': 't5_chat',
            'image_aspect_ratio': 'pad',
        },
    },
}


class CLIPT5Model(BaseVisualModel):
    def __init__(self, device: str = "cuda"):
        """

        Parameters
        ----------
        device: string that sets up torch device
        """

        self._device = device
        self._system_message = ("A chat between a curious user and an artificial intelligence assistant. "
                                "The assistant gives helpful, detailed, and polite answers to the user's questions.")
        self._question_template = 'Does this figure show "{}"? Please answer yes or no.'
        self._answer_template = "Yes"
        self._image_token = "<image>"
        self._image_token_index = -200

        self._processor = None
        self._model = None
        self._tokenizer = None

        self._context_len = 2048
        self._padding = -100

    def preprocess_inputs(self, images: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """

        Parameters
        ----------
        images: a list with images (renders of the generated 3D object) stored as torch tensors on the device;
        prompt_list: a list of prompts

        Returns
        -------
        stacked_images: a torch tensor with stacked input images
        tokenized_prompts: a torch tensor with tokenized input prompts
        """
        stacked_images = torch.stack(images, dim=0).to(self._device) / 255.0
        stacked_images = stacked_images.permute(0, 3, 1, 2).to(torch.float16)
        stacked_images = F.interpolate(stacked_images, size=(336, 336), mode="bicubic", align_corners=False)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1) * 3
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1) * 3
        normalize = transforms.Normalize(mean, std)
        stacked_images = normalize(stacked_images)

        return stacked_images

    @torch.no_grad()
    @torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    def forward(self, images: list[torch.Tensor], texts: list[str], question_template: str = "", answer_template: str = ""):
        """
        Function for inferencing the loaded model

        Parameters
        ----------
        images: list of torch.Tensors with stored images
        texts: list of input prompts stored as strings
        question_template: question template defined as a string [optional]
        answer_template: answer template defined as a string [optional]

        Returns
        -------
        lm_probs: torch.Tensor with clip scores
        """
        assert len(images) == len(texts), "Number of images and texts must match"

        # Turn "a photo of a dog" into
        # Q: "Does this figure show "a photo of a dog"? Please answer yes or no."
        # A: "Yes"
        if question_template != "":
            self._question_template = question_template
        if answer_template != "":
            self._answer_template = answer_template

        questions = [self._question_template.format(text) for text in texts]
        answers = [self._answer_template.format(text) for text in texts]

        # Formatting for CLIP-FlanT5 desired input including system message and image tokens
        questions = [self._format_question(question) for question in questions]
        answers = [self._format_answer(answer) for answer in answers]

        input_ids = [self._tokenize_image_token(qs, self._image_token_index, return_tensors='pt') for qs in questions]
        labels = [self._tokenize_image_token(ans, self._image_token_index, return_tensors='pt') for ans in answers]

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self._tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self._padding)

        input_ids = input_ids[:, :self._tokenizer.model_max_length]
        labels = labels[:, :self._tokenizer.model_max_length]

        attention_mask = input_ids.ne(self._tokenizer.pad_token_id)
        decoder_attention_mask = labels.ne(self._padding)

        # sending tensors to device
        input_ids = input_ids.to(self._device)
        attention_mask = attention_mask.to(self._device)
        decoder_attention_mask = decoder_attention_mask.to(self._device)
        labels = labels.to(self._device)

        images = self.preprocess_inputs(images)

        # model parameters
        model_input_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'decoder_attention_mask': decoder_attention_mask,
            'labels': labels,
            'images': images,
            'past_key_values': None,
            'inputs_embeds': None,
            'use_cache': None,
            'output_attentions': None,
            'output_hidden_states': None,
            'return_dict': True,
        }

        # inferencing model
        outputs = self._model(**model_input_kwargs)

        # unpacking values
        logits = outputs.logits
        lm_prob = torch.zeros(logits.shape[0])
        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')

        for k in range(lm_prob.shape[0]):
            # exp to cancel the log and get raw prob between 0 and 1
            lm_prob[k] = (-loss_fct(logits[k], labels[k])).exp()
        return lm_prob

    def _format_question(self, question: str):
        """
        Function for formating the question according pre-defined rules

        Parameters
        ----------
        question: input string with question-prompt template

        Returns
        -------
        formated question according to the predefined template
        """
        return self._system_message + " USER: " + self._image_token + "\n" + question + " ASSISTANT: "

    def _format_answer(self, answer):
        return answer

    def _tokenize_image_token(self, text: str, image_token_index: int, return_tensors: str | None = None):
        """
        Function for tokenizing input strings with image tokens

        Parameters
        ----------
        text: input list of strings that will be tokenized
        image_token_index: string that defines the image token
        return_tensors: 'pt' - pytorch tensors or None

        Returns
        -------
        input_ids: array or torch.Tensor with tokenized input text
        """
        text_chunks = [self._tokenizer(chunk).input_ids for chunk in text.split('<image>')]

        input_ids = []
        # Since there's no bos_token_id, simply concatenate the tokenized prompt_chunks with the image_token_index
        for x in self._insert_separator(text_chunks, [image_token_index]):
            input_ids.extend(x)

        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids

    @staticmethod
    def _insert_separator(X: list[int], sep: str):
        """
        Function for inserting a separator in the input string
        Parameters
        ----------
        X: input list that will be edited
        sep: separator to be inserted in the input string

        Returns
        -------
        string with inserted separators
        """
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    def preload_model(self, model_name: str):
        """
        Function for preloading model

        Parameters
        ----------
        model_name: the name of the model that will be loaded: clip-flant5-xxl (~11B) or clip-flant5-xl (~3B)
        """
        print("Loading model ...")
        model_max_length = CLIP_T5_MODELS[model_name]['tokenizer']['model_max_length']
        # image_aspect_ratio = CLIP_T5_MODELS[self.model_name]['model']['image_aspect_ratio']

        tokenizer_dict = {}
        tokenizer_dict['model_max_length'] = model_max_length

        self._tokenizer = AutoTokenizer.from_pretrained(CLIP_T5_MODELS[model_name]["tokenizer"]["path"], **tokenizer_dict)
        self._model = CLIPT5ForConditionalGeneration.from_pretrained(CLIP_T5_MODELS[model_name]["model"]["path"])
        # self._model.resize_token_embeddings(len(self._tokenizer))  # might be redundant

        if not self._model.get_vision_tower().is_loaded:
            self._model.get_vision_tower().load_model()

        self._model.to(self._device, dtype=torch.bfloat16)
        self._model.requires_grad_(False)
        self._model.eval()

        self._processor = self._model.get_vision_tower().image_processor
        print("Done.")

    def unload_model(self):
        """Function for unloading the model"""
        del self._model
        del self._tokenizer
        del self._processor

        self._model = None
        self._tokenizer = None
        self._processor = None

        gc.collect()
        torch.cuda.empty_cache()
