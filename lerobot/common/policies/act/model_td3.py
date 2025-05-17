#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Action Chunking Transformer Policy

As per Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (https://arxiv.org/abs/2304.13705).
The majority of changes here involve removing unused code, unifying naming, and adding helpful comments.
"""

import math
from collections import deque
from itertools import chain
from typing import Callable

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.act.modeling_act import ACTPolicy, ACT, ACTDecoder


class CriticPolicy(ACTPolicy):

    config_class = ACTConfig
    name = "critic"

    def __init__(
        self,
        config: ACTConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__(config, dataset_stats)
        self.model = Critic(config)

    def copy_from_actor(self, actor):
        self.model.copy_from_actor(actor.model)

    def q_forward(self, batch, actions):
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = [batch[key] for key in self.config.image_features]

        batch = self.normalize_targets(batch)
        q_value = self.model.q_forward(batch, actions)

        return q_value
    def forward(self, batch):
        pass
    def q1_forward(self, batch, actions):
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = [batch[key] for key in self.config.image_features]

        batch = self.normalize_targets(batch)
        q_value = self.model.q1_forward(batch, actions)

        return q_value
    def feature_extract_params(self):
        i = 0
        target_member = [
            'model.vae_encoder.',
            'model.vae_encoder_cls_embed.',
            'model.vae_encoder_robot_state_input_proj.',
            'model.vae_encoder_action_input_proj.',
            'model.vae_encoder_latent_output_proj.',
            'model.vae_encoder_pos_enc.',
            'model.backbone.',
            'model.encoder.',
            'model.encoder_robot_state_input_proj.',
            # 'model.encoder_env_state_input_proj',
            'model.encoder_latent_input_proj.',
            'model.encoder_img_feat_input_proj.',
            'model.encoder_1d_feature_pos_embed.',
            'model.encoder_cam_feat_pos_embed.',
        ]
        for ele in self.named_parameters():
            for key in target_member:
                if key in ele[0]:
                    yield ele[1]


class Critic(ACT):
    def __init__(
        self,
        config: ACTConfig,
        n_critics: int = 2,
    ):
        super().__init__(config)
        # critic与actor除decoder外共享参数，即共享vae_encode + vae_decode中的encoder
        # 另外,critic中有单独的q值的head
        self.decoder_actions_input_proj0 = nn.Linear(
            config.robot_state_feature.shape[0], config.dim_model
        )
        self.decoder_actions_input_proj1 = nn.Linear(
            config.robot_state_feature.shape[0], config.dim_model
        )
        self.decoder0 = ACTDecoder(config)
        self.decoder1 = ACTDecoder(config)
        #self._reset_parameters()
        self.q_head0 = nn.Sequential(  # 输出头
                nn.Linear(config.dim_model, 128),  # 聚合层
                nn.LayerNorm(128),
                nn.Linear(128, 1)
        )
        self.q_head1 = nn.Sequential(  # 输出头
            nn.Linear(config.dim_model, 128),  # 聚合层
            nn.LayerNorm(128),
            nn.Linear(128, 1)
        )
        self.decoder_pos_embed0 = nn.Embedding(config.chunk_size, config.dim_model)
        self.decoder_pos_embed1 = nn.Embedding(config.chunk_size, config.dim_model)

    def copy_from_actor(self, actor):
        self.vae_encoder = actor.vae_encoder
        self.vae_encoder_cls_embed = actor.vae_encoder_cls_embed
        self.vae_encoder_robot_state_input_proj = actor.vae_encoder_robot_state_input_proj
        self.vae_encoder_action_input_proj = actor.vae_encoder_action_input_proj
        self.vae_encoder_latent_output_proj = actor.vae_encoder_latent_output_proj
        self.vae_encoder_pos_enc = actor.vae_encoder_pos_enc
        self.backbone = actor.backbone
        self.encoder = actor.encoder
        self.encoder_robot_state_input_proj = actor.encoder_robot_state_input_proj
        #self.encoder_env_state_input_proj = actor.encoder_env_state_input_proj
        self.encoder_latent_input_proj = actor.encoder_latent_input_proj
        self.encoder_img_feat_input_proj = actor.encoder_img_feat_input_proj
        self.encoder_1d_feature_pos_embed = actor.encoder_1d_feature_pos_embed
        self.encoder_cam_feat_pos_embed = actor.encoder_cam_feat_pos_embed

    def q_forward_attn_out(self, batch):
        # 与actor共享大部分参数，所以不进行梯度更新，只在actor中更新
        with torch.no_grad():
            if self.config.use_vae and self.training:
                assert "action" in batch, (
                    "actions must be provided when using the variational objective in training mode."
                )

            if "observation.images" in batch:
                batch_size = batch["observation.images"][0].shape[0]
            else:
                batch_size = batch["observation.environment_state"].shape[0]

            # Prepare the latent for input to the transformer encoder.
            if self.config.use_vae and "action" in batch:
                # Prepare the input to the VAE encoder: [cls, *joint_space_configuration, *action_sequence].
                cls_embed = einops.repeat(
                    self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
                )  # (B, 1, D)
                if self.config.robot_state_feature:
                    robot_state_embed = self.vae_encoder_robot_state_input_proj(batch["observation.state"])
                    robot_state_embed = robot_state_embed.unsqueeze(1)  # (B, 1, D)
                action_embed = self.vae_encoder_action_input_proj(batch["action"])  # (B, S, D)

                if self.config.robot_state_feature:
                    vae_encoder_input = [cls_embed, robot_state_embed, action_embed]  # (B, S+2, D)
                else:
                    vae_encoder_input = [cls_embed, action_embed]
                vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

                # Prepare fixed positional embedding.
                # Note: detach() shouldn't be necessary but leaving it the same as the original code just in case.
                pos_embed = self.vae_encoder_pos_enc.clone().detach()  # (1, S+2, D)

                # Prepare key padding mask for the transformer encoder. We have 1 or 2 extra tokens at the start of the
                # sequence depending whether we use the input states or not (cls and robot state)
                # False means not a padding token.
                cls_joint_is_pad = torch.full(
                    (batch_size, 2 if self.config.robot_state_feature else 1),
                    False,
                    device=batch["observation.state"].device,
                )
                key_padding_mask = torch.cat(
                    [cls_joint_is_pad, batch["action_is_pad"]], axis=1
                )  # (bs, seq+1 or 2)

                # Forward pass through VAE encoder to get the latent PDF parameters.
                cls_token_out = self.vae_encoder(
                    vae_encoder_input.permute(1, 0, 2),
                    pos_embed=pos_embed.permute(1, 0, 2),
                    key_padding_mask=key_padding_mask,
                )[0]  # select the class token, with shape (B, D)
                latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
                mu = latent_pdf_params[:, : self.config.latent_dim]
                # This is 2log(sigma). Done this way to match the original implementation.
                log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]

                # Sample the latent with the reparameterization trick.
                latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
            else:
                # When not using the VAE encoder, we set the latent to be all zeros.
                mu = log_sigma_x2 = None
                # TODO(rcadene, alexander-soare): remove call to `.to` to speedup forward ; precompute and use buffer
                latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(
                    batch["observation.state"].device
                )

            # Prepare transformer encoder inputs.
            encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
            encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
            # Robot state token.
            if self.config.robot_state_feature:
                encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch["observation.state"]))
            # Environment state token.
            if self.config.env_state_feature:
                encoder_in_tokens.append(
                    self.encoder_env_state_input_proj(batch["observation.environment_state"])
                )

            # Camera observation features and positional embeddings.
            if self.config.image_features:
                all_cam_features = []
                all_cam_pos_embeds = []

                # For a list of images, the H and W may vary but H*W is constant.
                for img in batch["observation.images"]:
                    cam_features = self.backbone(img)["feature_map"]
                    cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                    cam_features = self.encoder_img_feat_input_proj(cam_features)

                    # Rearrange features to (sequence, batch, dim).
                    cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                    cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

                    all_cam_features.append(cam_features)
                    all_cam_pos_embeds.append(cam_pos_embed)

                encoder_in_tokens.extend(torch.cat(all_cam_features, axis=0))
                encoder_in_pos_embed.extend(torch.cat(all_cam_pos_embeds, axis=0))

            # Stack all tokens along the sequence dimension.
            encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
            encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

            # Forward pass through the transformer modules.
            encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
            # TODO(rcadene, alexander-soare): remove call to `device` ; precompute and use buffer
            #decoder_in = torch.zeros(
            #    (self.config.chunk_size, batch_size, self.config.dim_model),
            #    dtype=encoder_in_pos_embed.dtype,
            #    device=encoder_in_pos_embed.device,
            #)

            return encoder_out, encoder_in_pos_embed

    def q_forward(self, batch, actions):
        encoder_out, encoder_in_pos_embed = self.q_forward_attn_out(batch)
        decoder_in = torch.transpose(actions, 0, 1).contiguous()

        q0 = q1 = 0.0
        decoder_out = self.decoder(
            self.decoder_actions_input_proj0(decoder_in),
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed0.weight.unsqueeze(1)
        )
        # Move back to (B, S, C) then -> (B, C).
        decoder_out = decoder_out.transpose(0, 1).mean(dim=1)
        q0 = self.q_head0(decoder_out)
        decoder_out = self.decoder(
            self.decoder_actions_input_proj1(decoder_in),
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed1.weight.unsqueeze(1)
        )
        # Move back to (B, S, C) then -> (B, C).
        decoder_out = decoder_out.transpose(0, 1).mean(dim=1)
        q1 = self.q_head1(decoder_out)
        return (q0, q1)
    def q1_forward(self, batch, actions):
        encoder_out, encoder_in_pos_embed = self.q_forward_attn_out(batch)
        decoder_in = torch.transpose(actions, 0, 1).contiguous()

        q0 = 0.0
        decoder_out = self.decoder(
            self.decoder_actions_input_proj0(decoder_in),
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed0.weight.unsqueeze(1)
        )
        # Move back to (B, S, C) then -> (B, C).
        decoder_out = decoder_out.transpose(0, 1).mean(dim=1)
        q0 = self.q_head0(decoder_out)
        return q0
    def forward(self, batch):
        pass