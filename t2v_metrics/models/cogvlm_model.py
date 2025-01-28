import gc
import copy

import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer
from t2v_metrics.visual_model import BaseVisualModel


COGVLM_MODELS = {
    'cogvlm-17b': {
        'tokenizer' : {
            'path': 'lmsys/vicuna-7b-v1.5',
        },
        'model': {
            'path': 'THUDM/cogvlm-chat-hf',
            'conversation': 'chat',
            'image_aspect_ratio': 'pad',
        },
    },
}


class CogVLMModel(BaseVisualModel):
    """A wrapper for the CogVLM model"""
    def __init__(self, context_len: int = 2048):
        super(CogVLMModel, self).__init__()

        self._question_template = 'Does this figure show "{}"? Please answer yes or no.'
        self._answer_template = "Yes"

        self._model = None
        self._processor = None
        self._tokenizer: LlamaTokenizer = None

        self._context_len = context_len
        self._ignore_ind = -100

    def preload_model(self, model_name: str, torch_type: torch.dtype | None = None):
        """Load the model, tokenizer, image transform
        """

        self._model = AutoModelForCausalLM.from_pretrained(
            COGVLM_MODELS[model_name]['model']['path'],
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            )
        self._model.to(self._device)
        self._model.eval()
        self._tokenizer = LlamaTokenizer.from_pretrained(COGVLM_MODELS[model_name]['tokenizer']['path'], add_bos_token=False)

    def unload_model(self):
        del self._model
        del self._tokenizer

        torch.cuda.empty_cache()
        gc.collect()

        self._model = None
        self._tokenizer = None

    def format_question(self, question, conversation_style='chat'):
        if conversation_style == 'plain':
            question = question
        elif conversation_style == 'chat':
            question = question
        else:
            raise NotImplementedError()
        return question

    def format_answer(self, answer, conversation_style='chat'):
        if conversation_style == 'plain':
            answer = answer + "\n"
        elif conversation_style == 'chat':
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

        # Format questions and answers
        questions = [question_template.format(text) for text in texts]
        answers = [answer_template.format(text) for text in texts]
        
        questions = [self.format_question(question, conversation_style="chat") for question in questions]
        answers = [self.format_answer(answer, conversation_style="chat") for answer in answers]

        # Build input ids for the question
        q = self._model.build_conversation_input_ids(self._tokenizer, query=questions[0], history=[], images=[images[0]], template_version="chat")

        question_len = len(q['input_ids'])

        # Append token ids of the answer at the end
        tokens_to_append = self._tokenizer.encode(answers[0])
        q['input_ids']=torch.hstack([q['input_ids'],torch.tensor(tokens_to_append)])
        q['token_type_ids'] = torch.hstack([q['token_type_ids'],torch.tensor([0]*len(tokens_to_append))])
        q['attention_mask'] = torch.hstack([q['attention_mask'],torch.tensor([1]*len(tokens_to_append))])

        # Create labels ids and mask question tokens
        labels = copy.deepcopy(q['input_ids'])
        labels[:question_len] = self._ignore_ind

        # Prepare inputs and labels for model
        inputs = {
            'input_ids': q['input_ids'].unsqueeze(0).to('cuda'),
            'token_type_ids': q['token_type_ids'].unsqueeze(0).to('cuda'),
            'attention_mask': q['attention_mask'].unsqueeze(0).to('cuda'),
            'images': [[q['images'][0].to('cuda').to(torch.bfloat16)]],
            'labels': labels.unsqueeze(0).to('cuda'),
        }

        outputs = self._model(**inputs, return_dict=True)

        # Compute VQAScore from the loss
        loss = outputs.loss
        lm_prob = (-loss).exp()
        return lm_prob



    




