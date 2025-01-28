import copy
import gc
from PIL import Image

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer, Idefics3Model, IdeficsProcessor
from t2v_metrics.visual_model import BaseVisualModel


SMOLVLM_MODELS = {
    'smolvlm-1.7b-base': {
        'ckpt_path': 'HuggingFaceTB/SmolVLM-Base',
    },
    'smolvlm-1.7b-synth': {
        'ckpt_path': 'HuggingFaceTB/SmolVLM-Synthetic',
    },
}


class SmolVLMModel(BaseVisualModel):
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

        self._processor: IdeficsProcessor = None
        self._model: Idefics3Model = None
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
        self._model = AutoModelForVision2Seq.from_pretrained(
            SMOLVLM_MODELS[model_name]["ckpt_path"],
            torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2" if self._device == "cuda" else "eager")
        self._model.to(self._device)
        self._model.eval()

        self._processor = AutoProcessor.from_pretrained(SMOLVLM_MODELS[model_name]["ckpt_path"], size={"longest_edge": 2*384} )
        # self._tokenizer = self._processor.tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(SMOLVLM_MODELS[model_name]["ckpt_path"])

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

    def format_question(self, question: str):
        """

        Parameters
        ----------
        question

        Returns
        -------

        """
        # return {"type": "text", "text": question }
        system_msg = ("A chat between a curious user and an artificial intelligence assistant. "
                      "The assistant gives helpful, detailed, and polite answers to the user's questions.")
        question = "USER: " + "<image>" + "\n" + question + " ASSISTANT: "
        return question

    def format_answer(self, answer: str):
        """

        Parameters
        ----------
        answer

        Returns
        -------

        """

        # return {"type": "text", "text": answer}
        return answer + "</s>"

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
        content_imgs = []
        for i in range(num_imgs):
            content_imgs.append({"type": "image"})

        content_imgs.append( self.format_question(question))
        content_answer = self.format_answer("Yes")

        messages = [
            {
                "role": "user",
                "content": content_imgs,
            },
            # {
            #     "role": "assistant",
            #     "content": [content_answer],
            # }
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

        question = self.format_question(self._question_template.format(texts[0]))
        answer = self.format_answer(self._answer_template.format(texts[0]))

        # messages = self.create_message_template(len(images), question)
        # prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self._processor(text=question,
                                 images = images,
                                 return_tensors="pt",
                                 padding=True,
                                 )

        print(inputs)

        # inputs = inputs.to(self._device)
        question_len = len(inputs['input_ids'])

        # print(prompt)
        # inputs consists of combined data: imagery + textual that were tokenized and preprocessed during call to processor
        tokens_to_append = torch.tensor(self._tokenizer.encode(answer))
        # print(inputs['attention_mask'].shape, " / ", tokens_to_append.shape)

        inputs["input_ids"] = torch.hstack([inputs["input_ids"].squeeze(0), tokens_to_append]).unsqueeze(0)
        inputs["attention_mask"] = torch.hstack([inputs['attention_mask'].squeeze(0), torch.tensor([1]*len(tokens_to_append))]).unsqueeze(0)

        labels = copy.deepcopy(inputs["input_ids"])

        # setting image tokens to negative value, they will be ignored during inference
        # labels[labels==self._model.image_token_id] = self._padding
        labels[:question_len] = self._padding

        # print(labels)
        # print(self._model.image_token_id)
        # print(self._model.config.vocab_size)

        model_input_kwargs = {
            'input_ids': inputs["input_ids"].to(self._device),
            'pixel_values': inputs["pixel_values"].to(self._device),
            'attention_mask': inputs["attention_mask"].to(self._device),
            'pixel_attention_mask': inputs["pixel_attention_mask"].to(self._device),
            'return_dict': True,
            'labels': labels.to(self._device),
        }

        # model_input_kwargs = inputs

        outputs = self._model(**model_input_kwargs)
        # print(outputs)

        # generated_ids = self._model.generate(**model_input_kwargs)
        # generated_texts = self._processor.batch_decode(
        #     generated_ids,
        #     skip_special_tokens=True,
        # )
        # print(generated_ids)

        # print(generated_texts[0])
        # print(type(outputs))
        # print(outputs)
        # print(outputs.keys())

        # unpacking values
        # loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        # output_logits = outputs.logits[:, :-1, :].contiguous()
        # output_labels = labels[:, 1:].contiguous().to(self._device)

        # print(labels, "\n\n")
        # print(output_labels)


        # lm_prob = torch.zeros(output_logits.shape[0])
        # for i in range(lm_prob.shape[0]):
        #     lm_prob[i] = (-loss_fct(output_logits[i], output_labels[i])).exp()

        loss = outputs.loss
        lm_prob = (-loss).exp()

        return lm_prob