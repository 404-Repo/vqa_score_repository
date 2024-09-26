#    Copyright 2023 Zhiqiu Lin
#    Copyright 2024 Alexander Tereshin (ctranslate2 implementation)
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from typing import List, Optional, Tuple, Union
from dataclasses import dataclass, field

import torch
from transformers import AutoConfig, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import Seq2SeqLMOutput
# from transformers import T5Config, T5ForConditionalGeneration

from turbot5 import T5ForConditionalGeneration
from turbot5 import T5Config

from ..multimodal_encoder.builder import build_vision_tower
from ..multimodal_projector.builder import build_vision_projector


@dataclass
class ModelArguments:
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default='openai/clip-vit-large-patch14-336')
    mm_vision_select_layer: Optional[int] = field(default=-2)   # default to the second last layer in llava1.5
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='mlp2x_gelu')
    mm_vision_select_feature: Optional[str] = field(default="patch")


class CLIPT5Config(T5Config):
    model_type = "clip_t5"


class CLIPT5ForConditionalGeneration(T5ForConditionalGeneration):
    # This class supports both T5 and FlanT5
    config_class = CLIPT5Config
    IMAGE_TOKEN_INDEX = -200

    def __init__(self, config):
        super(CLIPT5ForConditionalGeneration, self).__init__(config)
        self.embed_tokens = self.encoder.embed_tokens
        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config)
            self.mm_projector = build_vision_projector(config)

    def get_vision_tower(self):
        """"""
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_model(self):
        """"""
        return self

    def prepare_inputs_labels_for_multimodal(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, images: torch.Tensor):
        """

        Parameters
        ----------
        input_ids
        attention_mask
        images

        Returns
        -------

        """

        # The labels are now separated from the input_ids.
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            raise NotImplementedError()

        # encoding images stored as torch.Tensor
        image_features = self.encode_images(images)

        new_input_embeds = []
        cur_image_idx = 0

        for cur_input_ids in input_ids:
            # Check if the current sample is multimodal
            image_token_indices = torch.where(cur_input_ids == self.IMAGE_TOKEN_INDEX)[0]

            if image_token_indices.numel() == 0:
                raise NotImplementedError("The current sample is not multimodal.")

            cur_new_input_embeds = []
            prev_index = 0  # To track where we left off after embedding tokens before an image

            for image_token_start in image_token_indices:
                # Append token embeddings up to the next image token
                if image_token_start > prev_index:
                    cur_new_input_embeds.append(self.embed_tokens(cur_input_ids[prev_index:image_token_start]))

                # Append image feature
                cur_new_input_embeds.append(image_features[cur_image_idx])
                cur_image_idx += 1
                prev_index = image_token_start + 1

            # Embed any remaining tokens after the last image token
            if prev_index < cur_input_ids.size(0):
                cur_new_input_embeds.append(self.embed_tokens(cur_input_ids[prev_index:]))

            # Move embeddings to the correct device and concatenate them
            cur_new_input_embeds = torch.cat([x.to(device=self.device) for x in cur_new_input_embeds], dim=0)
            new_input_embeds.append(cur_new_input_embeds)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            _input_embeds_lengths = [embed.shape[0] for embed in new_input_embeds]

            # Create the tensor for aligned embeddings directly
            padded_embeds = [
                torch.cat([embed, torch.zeros((max_len - embed.shape[0], embed.shape[1]),
                                              dtype=embed.dtype, device=embed.device)], dim=0)
                if embed.shape[0] < max_len else embed
                for embed in new_input_embeds
            ]

            # Stack into a single tensor
            new_input_embeds = torch.stack(padded_embeds, dim=0)

            if attention_mask is not None:
                new_attention_mask = [
                    torch.cat([
                        torch.full((_input_embeds_length - input_ids.shape[1],), True,
                                   dtype=attention_mask.dtype, device=attention_mask.device),
                        cur_attention_mask,
                        torch.full((new_input_embeds.shape[1] - _input_embeds_length,), False,
                                   dtype=attention_mask.dtype, device=attention_mask.device)
                    ], dim=0)
                    for cur_attention_mask, _input_embeds_length in zip(attention_mask, _input_embeds_lengths)
                ]

                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full(
                    (attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, new_input_embeds

    def encode_images(self, images: torch.Tensor):
        """

        Parameters
        ----------
        images

        Returns
        -------

        """
        image_features = self.get_vision_tower()(images)
        image_features = self.mm_projector(image_features)
        return image_features

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        """

        Parameters
        ----------
        input_ids
        attention_mask
        decoder_attention_mask
        past_key_values
        inputs_embeds
        labels
        use_cache
        output_attentions
        output_hidden_states
        images
        return_dict
        kwargs

        Returns
        -------

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            _, attention_mask, inputs_embeds = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, images)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = super(CLIPT5ForConditionalGeneration, self).forward(
            input_ids=None, # will be None if inputs_embeds is not None
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        return outputs

    @torch.no_grad()
    def generate(
            self,
            inputs: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            images: Optional[torch.Tensor] = None,
            **kwargs,
    ):
        """

        Parameters
        ----------
        inputs
        attention_mask
        images
        kwargs

        Returns
        -------

        """
        assert images is not None, "images must be provided"
        assert inputs is not None, "inputs must be provided"
        assert attention_mask is not None, "attention_mask must be provided"
        _, attention_mask, inputs_embeds, = self.prepare_inputs_labels_for_multimodal(inputs, attention_mask, images)
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = super(CLIPT5ForConditionalGeneration, self).generate(
            input_ids=None,  # will be None if inputs_embeds is not None
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )
        return outputs

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            decoder_attention_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            inputs_embeds=None,
            **kwargs,
    ):
        """

        Parameters
        ----------
        input_ids
        past_key_values
        attention_mask
        head_mask
        decoder_head_mask
        decoder_attention_mask
        cross_attn_head_mask
        use_cache
        encoder_outputs
        inputs_embeds
        kwargs

        Returns
        -------

        """
        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update({
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        })
        return model_inputs


AutoConfig.register("clip_t5", CLIPT5Config)
AutoModelForSeq2SeqLM.register(CLIPT5Config, CLIPT5ForConditionalGeneration)
