import re
import time
import torch
import torch.nn.functional as F
from logzero import logger
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor, DynamicCache
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb

from inference.abstract_hermes import Abstract_Hermes
from inference.reindex_3d import (
    get_cache_seq_len,
    contiguous_kv,
    _get_rotary_module,
    _get_mrope_section,
    compute_cos_sin_for_positions,
    rotary_delta,
    apply_rotary_delta_to_keys_only,
)


import transformers.modeling_flash_attention_utils

if not hasattr(transformers.modeling_flash_attention_utils, "_original_prepare_fa_kwargs"):
    transformers.modeling_flash_attention_utils._original_prepare_fa_kwargs = transformers.modeling_flash_attention_utils.prepare_fa_kwargs_from_position_ids

    def _patched_prepare_fa_kwargs(position_ids, attention_mask=None):
        if position_ids is not None and position_ids.dim() == 3:
            batch_size = position_ids.shape[1]
            seq_len = position_ids.shape[2]
            device = position_ids.device
            cu_seq_lens = torch.arange(
                0, (batch_size + 1) * seq_len, step=seq_len,
                dtype=torch.int32, device=device
            )
            max_length = seq_len
            return (cu_seq_lens, cu_seq_lens), (max_length, max_length)
        return transformers.modeling_flash_attention_utils._original_prepare_fa_kwargs(position_ids, attention_mask)

    transformers.modeling_flash_attention_utils.prepare_fa_kwargs_from_position_ids = _patched_prepare_fa_kwargs


def get_qwen2_5_vl_position_ids(video_grid_thw, seq_len, offset=0, spatial_merge_size=2, vision_config=None, sample_fps=1):
    t, h, w = video_grid_thw
    llm_grid_t = t
    llm_grid_h = h // spatial_merge_size
    llm_grid_w = w // spatial_merge_size

    if vision_config is not None:
        spatial_merge_size = vision_config.spatial_merge_size
        llm_grid_h = h // spatial_merge_size
        llm_grid_w = w // spatial_merge_size

    if llm_grid_t * llm_grid_h * llm_grid_w != seq_len:
        if t * h * w == seq_len:
            llm_grid_t, llm_grid_h, llm_grid_w = t, h, w
        else:
             print(f"Warning: Grid {t}x{h}x{w} (merged: {llm_grid_t}x{llm_grid_h}x{llm_grid_w}) does not match seq_len {seq_len}")

    if vision_config is not None:
        tokens_per_second = vision_config.tokens_per_second
        second_per_grid_t = 2 / sample_fps
        range_tensor = torch.arange(llm_grid_t, dtype=torch.float32).view(-1, 1)
        expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)
        time_tensor = expanded_range * second_per_grid_t * tokens_per_second
        t_index = time_tensor.flatten() + offset
    else:
        t_index = torch.arange(llm_grid_t, dtype=torch.float32).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten() + offset

    h_index = torch.arange(llm_grid_h, dtype=torch.float32).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten() + offset
    w_index = torch.arange(llm_grid_w, dtype=torch.float32).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten() + offset

    return torch.stack([t_index, h_index, w_index])


class QwenVL_Hermes(Qwen2_5_VLForConditionalGeneration, Abstract_Hermes):
    """
    Qwen2.5-VL with HERMES strict-shrink support.
    Uses 3D M-RoPE (Multimodal RoPE) and inter-layer consistency optimization.
    """

    def __init__(self, config, processor, init_prompt_ids, kv_size, streaming=True, sample_fps=1):
        Abstract_Hermes.__init__(self, processor, init_prompt_ids, kv_size)
        self.streaming = streaming
        self.sample_fps = sample_fps

        num_layers = config.num_hidden_layers if hasattr(config, 'num_hidden_layers') else 28
        self.num_layers = num_layers

        self.short_term_ratio = 0.3
        self.long_term_ratio = 0.3
        self.short_term_threshold = int(self.num_layers * self.short_term_ratio)
        self.long_term_threshold = int(self.num_layers * (1 - self.long_term_ratio))

        self._position_ids_cache = [None for _ in range(num_layers)]

        self._layer_position_ids = {}
        self._hook_handles = []

        self.total_processed_frames = 0

        self._mrope_section = _get_mrope_section(self.language_model)

        self._register_forward_hooks()

    def _ensure_dynamic_cache(self):
        if self.kv_cache is None:
            return
        if not isinstance(self.kv_cache, DynamicCache):
            self.kv_cache = DynamicCache.from_legacy_cache(self.kv_cache)

    def _sanitize_keep_indices(self, keep_indices_1d: torch.Tensor, seq_len: int) -> torch.Tensor:
        keep_indices_1d = keep_indices_1d.to(self.device)
        keep_indices_1d = keep_indices_1d[(keep_indices_1d >= 0) & (keep_indices_1d < seq_len)]
        if keep_indices_1d.numel() == 0:
            return torch.tensor([0], device=self.device)
        keep_indices_1d = torch.unique(keep_indices_1d, sorted=True)
        return keep_indices_1d

    def _register_forward_hooks(self):
        def make_hook(layer_idx):
            def hook(module, args, kwargs):
                if layer_idx in self._layer_position_ids:
                    kwargs['position_ids'] = self._layer_position_ids[layer_idx]
                return args, kwargs
            return hook

        for layer_idx, layer in enumerate(self.language_model.layers):
            handle = layer.register_forward_pre_hook(make_hook(layer_idx), with_kwargs=True)
            self._hook_handles.append(handle)

    def _clear_forward_hooks(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

    def _append_position_ids_layer(self, layer_idx: int, start_per_dim: list, length: int):
        device = self.device
        new_pos = torch.zeros((3, length), device=device, dtype=torch.float32)
        for dim in range(3):
            new_pos[dim, :] = torch.arange(start_per_dim[dim], start_per_dim[dim] + length, device=device, dtype=torch.float32)

        if self._position_ids_cache[layer_idx] is None:
            self._position_ids_cache[layer_idx] = new_pos
        else:
            self._position_ids_cache[layer_idx] = torch.cat(
                [self._position_ids_cache[layer_idx], new_pos], dim=1
            )

    def _append_position_ids_layer_explicit(self, layer_idx: int, new_pos: torch.Tensor):
        if self._position_ids_cache[layer_idx] is None:
            self._position_ids_cache[layer_idx] = new_pos.to(self.device)
        else:
            self._position_ids_cache[layer_idx] = torch.cat(
                [self._position_ids_cache[layer_idx], new_pos.to(self.device)], dim=1
            )

    def _get_cache_seq_len_per_layer(self) -> list:
        if self.kv_cache is None:
            return [0] * self.num_layers

        lengths = []
        for layer_idx in range(len(self.kv_cache)):
            k_layer, v_layer = self.kv_cache[layer_idx]
            lengths.append(k_layer.shape[2])
        return lengths

    def _get_next_global_offset_per_layer(self) -> list:
        offsets = []
        for layer_idx in range(self.num_layers):
            if (layer_idx < len(self._position_ids_cache) and
                self._position_ids_cache[layer_idx] is not None and
                self._position_ids_cache[layer_idx].numel() > 0):
                cache = self._position_ids_cache[layer_idx]
                global_max = cache.max().item()
                offsets.append(int(global_max) + 1)
            else:
                offsets.append(0)
        return offsets

    def _build_position_ids_3d_for_text(self, global_offset: int, q_len: int, batch: int) -> torch.Tensor:
        """[3, batch, q_len] -- text tokens use identical position across all 3 dims."""
        pos_1d = torch.arange(global_offset, global_offset + q_len, device=self.device, dtype=torch.float32)
        pos_2d = pos_1d.unsqueeze(0).expand(batch, -1)
        pos_3d = pos_2d.unsqueeze(0).expand(3, -1, -1).clone()
        return pos_3d

    def _build_position_ids_3d_for_vision(self, grid_pos_ids: torch.Tensor, batch: int) -> torch.Tensor:
        """[3, batch, q_len] -- vision tokens use grid-based 3D positions."""
        pos_3d = grid_pos_ids.unsqueeze(1).expand(3, batch, -1).clone()
        return pos_3d

    @torch.inference_mode()
    def _shrink_positions_and_rerotate_keys(self, keep_indices_per_layer):
        device = self.device
        curr_lens = self._get_cache_seq_len_per_layer()

        for layer_idx in range(self.num_layers):
            layer_len = curr_lens[layer_idx] if layer_idx < len(curr_lens) else curr_lens[0]
            if (self._position_ids_cache[layer_idx] is None or
                self._position_ids_cache[layer_idx].shape[1] != layer_len):
                pos = torch.arange(layer_len, device=device, dtype=torch.float32)
                self._position_ids_cache[layer_idx] = pos.unsqueeze(0).expand(3, -1).clone()

        max_pos_limit = getattr(self.language_model.config, "max_position_embeddings", 128000)
        compact_threshold = max_pos_limit - 1024

        current_max_pos = 0
        for layer_cache in self._position_ids_cache:
            if layer_cache is not None and layer_cache.numel() > 0:
                current_max_pos = max(current_max_pos, layer_cache.max().item())

        should_compact = (current_max_pos > compact_threshold) if self.streaming else True

        if should_compact:
            logger.info(f"[Shrink] Max position {current_max_pos} > {compact_threshold}. Compacting position IDs (Shift Left).")

        old_position_ids_cache = [cache.clone() if cache is not None else None for cache in self._position_ids_cache]

        sample_k = self.kv_cache[0][0]
        dtype = sample_k.dtype
        mrope_section = self._mrope_section

        new_kv_cache = []

        for layer_idx, (k_layer, v_layer) in enumerate(self.kv_cache):
            keep_indices_layer = keep_indices_per_layer[layer_idx]
            if not isinstance(keep_indices_layer, torch.Tensor):
                keep_indices_layer = torch.as_tensor(keep_indices_layer, device=device)

            seq_len_layer = k_layer.shape[2]
            safe_idx = self._sanitize_keep_indices(keep_indices_layer, seq_len_layer)

            if safe_idx.numel() == 0:
                logger.warning(f"Layer {layer_idx}: After sanitization, keep_indices is empty; keeping first token")
                safe_idx = torch.tensor([0], device=device)

            is_long_term = (layer_idx >= self.long_term_threshold)

            k_kept = torch.index_select(k_layer, dim=2, index=safe_idx)
            v_kept = torch.index_select(v_layer, dim=2, index=safe_idx)
            old_pos_kept = old_position_ids_cache[layer_idx][:, safe_idx]

            if should_compact:
                new_pos_kept = old_pos_kept.clone()
                if new_pos_kept.shape[1] > 0:
                    text_offset = self.visual_start_idx
                    num_text_kept = (safe_idx < text_offset).sum().item()
                    num_video_kept = (safe_idx >= text_offset).sum().item()

                    if num_video_kept > 0:
                        video_indices_in_kept = torch.arange(num_text_kept, new_pos_kept.shape[1], device=device)
                        old_video_pos = old_pos_kept[:, video_indices_in_kept]
                        for dim in range(3):
                            old_vals = old_video_pos[dim]
                            unique_vals, inverse_indices = torch.unique(old_vals, sorted=True, return_inverse=True)
                            compact_map = torch.arange(len(unique_vals), device=device) + text_offset
                            new_pos_kept[dim, video_indices_in_kept] = compact_map[inverse_indices].to(new_pos_kept.dtype)

                cos_old, sin_old = compute_cos_sin_for_positions(
                    self.language_model, len(safe_idx), old_pos_kept, dtype, device
                )
                cos_new, sin_new = compute_cos_sin_for_positions(
                    self.language_model, len(safe_idx), new_pos_kept, dtype, device
                )
                cos_delta, sin_delta = rotary_delta(cos_old, sin_old, cos_new, sin_new)

                try:
                    k_kept_final = apply_rotary_delta_to_keys_only(k_kept, cos_delta, sin_delta, mrope_section)
                except Exception as e:
                    logger.error(f"apply_rotary_delta failed at layer {layer_idx}: "
                                 f"k_kept={tuple(k_kept.shape)}, cos_delta={tuple(cos_delta.shape)}, err={e}")
                    raise
            else:
                new_pos_kept = old_pos_kept
                k_kept_final = k_kept

            if is_long_term:
                mask = torch.ones(seq_len_layer, dtype=torch.bool, device=device)
                mask[safe_idx] = False
                prune_indices = torch.nonzero(mask).squeeze(1)

                if prune_indices.numel() > 0:
                    k_pruned = torch.index_select(k_layer, dim=2, index=prune_indices)
                    v_pruned = torch.index_select(v_layer, dim=2, index=prune_indices)

                    v_summary = v_pruned.mean(dim=2, keepdim=True)

                    old_pos_pruned = old_position_ids_cache[layer_idx][:, prune_indices]

                    summary_pos_id = new_pos_kept.max().item() + 1
                    summary_pos_tensor = torch.tensor([summary_pos_id], device=device, dtype=torch.float32).repeat(3, 1)
                    target_pos_pruned = summary_pos_tensor.expand(3, old_pos_pruned.shape[1])

                    cos_old, sin_old = compute_cos_sin_for_positions(
                        self.language_model, old_pos_pruned.shape[1], old_pos_pruned, dtype, device
                    )
                    cos_new, sin_new = compute_cos_sin_for_positions(
                        self.language_model, target_pos_pruned.shape[1], target_pos_pruned, dtype, device
                    )
                    cos_delta, sin_delta = rotary_delta(cos_old, sin_old, cos_new, sin_new)

                    k_pruned_aligned = apply_rotary_delta_to_keys_only(k_pruned, cos_delta, sin_delta, mrope_section)
                    k_summary_final = k_pruned_aligned.mean(dim=2, keepdim=True)

                    k_final = torch.cat([k_kept_final, k_summary_final], dim=2)
                    v_final = torch.cat([v_kept, v_summary], dim=2)
                    new_pos_layer = torch.cat([new_pos_kept, summary_pos_tensor], dim=1)
                else:
                    k_final = k_kept_final
                    v_final = v_kept
                    new_pos_layer = new_pos_kept
            else:
                k_final = k_kept_final
                v_final = v_kept
                new_pos_layer = new_pos_kept

            new_kv_cache.append((k_final.contiguous(), v_final.contiguous()))
            self._position_ids_cache[layer_idx] = new_pos_layer.clone()

        self.kv_cache = new_kv_cache
        contiguous_kv(self.kv_cache)

        new_lens = self._get_cache_seq_len_per_layer()
        logger.info(f"Layer-wise shrink completed. Lengths: min={min(new_lens)}, max={max(new_lens)}, "
                   f"first={new_lens[0]}, last={new_lens[-1]}")

    @torch.inference_mode()
    def predict_next_question(self):
        if hasattr(self, 'conv_history') and len(self.conv_history) > 0:
            last_q, last_a, last_options = self.conv_history[-1]

            option_match = re.match(r'^\s*(?:\()?([A-Z])(?:\))?\.?\s*$', last_a)
            if option_match and last_options is not None:
                option_char = option_match.group(1)
                option_idx = ord(option_char) - ord('A')
                if 0 <= option_idx < len(last_options):
                    last_a = last_options[option_idx]

            last_a = re.sub(r'[A-Z]\) ', '', last_a)

            last_round_history = f"Question: {last_q} Answer: {last_a}"

            global_query = f"Context summary: {last_round_history}. Summarize the video narrative, identifying main characters, key events, timeline changes, and the overall theme."
            local_query = f"Find recent details related to: {last_round_history}. Describe the current scene in detail, focusing on specific objects, fine-grained actions, and spatial relationships."
        else:
            global_query = (
                "Summarize the video narrative, identifying main characters, key events, timeline changes, and the overall theme."
            )
            local_query = (
                "Describe the current scene in detail, focusing on specific objects, fine-grained actions, and spatial relationships."
            )

        print(f"Local question: {local_query}")
        print(f"Global question: {global_query}")
        return local_query, global_query

    @torch.inference_mode()
    def encode_init_prompt(self):
        if not isinstance(self.init_prompt_ids, torch.Tensor):
            self.init_prompt_ids = torch.as_tensor(self.init_prompt_ids, device=self.device)

        seq_len = self.init_prompt_ids.shape[-1]
        pos_1d = torch.arange(seq_len, device=self.device, dtype=torch.float32)
        position_ids_3d = pos_1d.unsqueeze(0).unsqueeze(0).expand(3, 1, -1).clone()

        output = self.language_model(
            input_ids=self.init_prompt_ids,
            use_cache=True,
            return_dict=True,
            position_ids=position_ids_3d,
        )
        self.kv_cache = output.past_key_values
        self.visual_start_idx = self.kv_cache[0][0].shape[2]

        self._ensure_dynamic_cache()
        self.total_processed_frames = 0

        curr_lens = self._get_cache_seq_len_per_layer()
        for layer_idx in range(self.num_layers):
            pos = torch.arange(curr_lens[layer_idx], device=self.device, dtype=torch.float32)
            self._position_ids_cache[layer_idx] = pos.unsqueeze(0).expand(3, -1).clone()

    @torch.inference_mode()
    def encode_video_chunk(self, video_chunk):
        if video_chunk is None or (hasattr(video_chunk, "shape") and video_chunk.shape[0] == 0):
            return

        if len(video_chunk.shape) == 4 and video_chunk.shape[-1] == 3:
            video_chunk = video_chunk.permute(0, 3, 1, 2)

        video_input = self.processor(text=[""], videos=video_chunk, return_tensors="pt").to(self.device, self.dtype)
        pixel_values_videos = video_input["pixel_values_videos"]
        video_grid_thw = video_input["video_grid_thw"]
        video_features = self.get_video_features(pixel_values_videos, video_grid_thw)[0].unsqueeze(0)

        self._ensure_dynamic_cache()

        global_offset_per_layer = self._get_next_global_offset_per_layer()
        q_len = video_features.shape[1]
        batch = video_features.shape[0]

        base_offset = global_offset_per_layer[0]
        grid_pos_ids = get_qwen2_5_vl_position_ids(
            video_grid_thw[0].tolist(),
            q_len,
            offset=base_offset,
            vision_config=self.config.vision_config,
            sample_fps=self.sample_fps,
        ).to(self.device)

        self._layer_position_ids.clear()
        for layer_idx in range(self.num_layers):
            layer_offset = global_offset_per_layer[layer_idx]
            current_layer_pos = grid_pos_ids.clone()
            if layer_offset != base_offset:
                current_layer_pos = current_layer_pos + (layer_offset - base_offset)
            position_ids_3d = self._build_position_ids_3d_for_vision(current_layer_pos, batch)
            self._layer_position_ids[layer_idx] = position_ids_3d

        default_position_ids_3d = self._build_position_ids_3d_for_vision(grid_pos_ids, batch)

        out = self.language_model(
            inputs_embeds=video_features,
            past_key_values=self.kv_cache,
            use_cache=True,
            return_dict=True,
            position_ids=default_position_ids_3d,
        )
        self.kv_cache = out.past_key_values
        contiguous_kv(self.kv_cache)

        for layer_idx in range(self.num_layers):
            layer_offset = global_offset_per_layer[layer_idx]
            current_layer_pos = grid_pos_ids.clone()
            if layer_offset != base_offset:
                current_layer_pos = current_layer_pos + (layer_offset - base_offset)
            self._append_position_ids_layer_explicit(layer_idx, current_layer_pos)

        self.last_encoded_frames = video_chunk.shape[0]
        self.total_processed_frames += video_chunk.shape[0]

        self._layer_position_ids.clear()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def apply_kv_cache_pruning_strict(self, keep_indices_all_layers):
        if self.kv_cache is None:
            logger.warning("No KV-Cache to prune")
            return
        if not keep_indices_all_layers or len(keep_indices_all_layers[0]) == 0:
            logger.warning("Empty keep_indices; skip pruning")
            return

        self._shrink_positions_and_rerotate_keys(keep_indices_all_layers)
        logger.info(f"Strict-shrunk KV cache. New length: {get_cache_seq_len(self.kv_cache)}")

    def allocate_budget_by_depth(self, total_budget, num_layers):
        budget_per_layer = [total_budget // num_layers] * num_layers
        diff = total_budget - sum(budget_per_layer)
        budget_per_layer[-1] += diff
        return budget_per_layer

    def _compute_attention_scores_manually(self, input_ids, past_key_values):
        device = self.device
        global_offset_per_layer = self._get_next_global_offset_per_layer()
        q_len = input_ids.shape[1]
        batch = input_ids.shape[0]

        inputs_embeds = self.get_input_embeddings()(input_ids)

        config = self.language_model.config
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        head_dim = config.hidden_size // num_heads
        hidden_size = config.hidden_size

        attention_weights_list = []
        hidden_states = inputs_embeds

        for layer_idx in range(num_layers):
            layer = self.language_model.layers[layer_idx]
            past_k, past_v = past_key_values[layer_idx]

            layer_offset = global_offset_per_layer[layer_idx]
            position_ids_3d = torch.zeros((3, 1, q_len), device=device, dtype=torch.float32)
            for dim in range(3):
                position_ids_3d[dim, 0, :] = torch.arange(layer_offset, layer_offset + q_len, device=device)

            hidden_states_norm = layer.input_layernorm(hidden_states)

            attn = layer.self_attn

            query_states = attn.q_proj(hidden_states_norm)
            query_states = query_states.view(batch, q_len, num_heads, head_dim).transpose(1, 2)

            key_states = attn.k_proj(hidden_states_norm)
            key_states = key_states.view(batch, q_len, num_key_value_heads, head_dim).transpose(1, 2)

            value_states = attn.v_proj(hidden_states_norm)
            value_states = value_states.view(batch, q_len, num_key_value_heads, head_dim).transpose(1, 2)

            rotary_emb = _get_rotary_module(self.language_model)
            dummy_h = torch.zeros((1, q_len, hidden_size), device=device, dtype=hidden_states.dtype)
            cos, sin = rotary_emb(dummy_h, position_ids_3d)

            query_states, key_states = apply_multimodal_rotary_pos_emb(
                query_states, key_states, cos, sin, self._mrope_section
            )

            key_states = torch.cat([past_k, key_states], dim=2)
            value_states = torch.cat([past_v, value_states], dim=2)

            if num_key_value_heads != num_heads:
                n_rep = num_heads // num_key_value_heads
                key_states = torch.repeat_interleave(key_states, n_rep, dim=1)
                value_states = torch.repeat_interleave(value_states, n_rep, dim=1)

            attn_weights = torch.matmul(query_states.float(), key_states.float().transpose(-2, -1)) / (head_dim ** 0.5)
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float16).to(query_states.dtype)
            attention_weights_list.append(attn_weights)

        return attention_weights_list

    def prune_kv_cache_by_attention(self, attn_weights_local, attn_weights_global, attn_weights_mixed, num_keep=3000):
        device = self.device
        visual_start_idx = self.visual_start_idx
        num_layers = len(attn_weights_local)

        question_len_local = attn_weights_local[0].shape[2]
        question_len_global = attn_weights_global[0].shape[2]
        question_len_mixed = attn_weights_mixed[0].shape[2]

        total_budget = num_keep * num_layers
        budget_per_layer = self.allocate_budget_by_depth(total_budget, num_layers)

        keep_indices_all_layers = []

        layer_raw_scores = []
        layer_configs = []

        for layer_idx in range(len(attn_weights_local)):
            if layer_idx < self.short_term_threshold:
                layer_type = "short-term"
                layer_attn_weights = attn_weights_local[layer_idx]
                question_len = question_len_local
                layer_recency_alpha = 1
                k = 20

            elif layer_idx >= self.long_term_threshold:
                layer_type = "long-term"
                layer_attn_weights = attn_weights_global[layer_idx]
                question_len = question_len_global
                layer_recency_alpha = 0
                k = 0.0

            else:
                layer_type = "mid-term"
                layer_attn_weights = attn_weights_mixed[layer_idx]
                question_len = question_len_mixed
                progress = (layer_idx - self.short_term_threshold) / (self.long_term_threshold - self.short_term_threshold)
                layer_recency_alpha = 0.75 - 0.6 * progress
                k = 20 - 12 * progress

            visual_attn_weights = layer_attn_weights[0].mean(dim=0)[:,visual_start_idx:-1*question_len].mean(dim=0)
            num_visual_tokens = visual_attn_weights.shape[0]
            layer_budget = budget_per_layer[layer_idx]

            if layer_type == 'long-term':
                layer_budget = max(0, layer_budget - 1)

            positions = torch.arange(num_visual_tokens, device=device, dtype=torch.float32)
            time_distances = (num_visual_tokens - 1 - positions) / max(num_visual_tokens - 1, 1)

            recency_weights = torch.exp(-k * time_distances)

            attn_norm = (visual_attn_weights - visual_attn_weights.min()) / \
                        (visual_attn_weights.max() - visual_attn_weights.min() + 1e-6)
            recency_norm = (recency_weights - recency_weights.min()) / \
                        (recency_weights.max() - recency_weights.min() + 1e-6)

            raw_score = attn_norm * (1 - layer_recency_alpha) + recency_norm * layer_recency_alpha

            layer_raw_scores.append(raw_score)
            layer_configs.append({
                'budget': min(layer_budget, num_visual_tokens),
                'layer_type': layer_type,
                'visual_start_idx': visual_start_idx
            })

        refined_scores = [s.clone() for s in layer_raw_scores]

        for i in range(len(refined_scores) - 2, -1, -1):
            current_type = layer_configs[i]['layer_type']

            if current_type == 'long-term':
                gamma = 0.4
            elif current_type == 'mid-term':
                gamma = 0.3
            else:
                gamma = 0.1

            score_current = refined_scores[i]
            score_next = refined_scores[i+1]

            if score_current.shape[0] != score_next.shape[0]:
                score_next_reshaped = score_next.view(1, 1, -1)
                score_next_interp = F.interpolate(
                    score_next_reshaped,
                    size=score_current.shape[0],
                    mode='linear',
                    align_corners=False
                ).view(-1)
                refined_scores[i] = (1 - gamma) * score_current + gamma * score_next_interp
            else:
                refined_scores[i] = (1 - gamma) * score_current + gamma * score_next

        for layer_idx, score in enumerate(refined_scores):
            config = layer_configs[layer_idx]
            actual_num_keep = config['budget']
            start_idx = config['visual_start_idx']

            topk_indices_relative = torch.topk(score, actual_num_keep, sorted=False)[1]
            topk_indices_absolute = topk_indices_relative + start_idx
            topk_indices_absolute_sorted = torch.sort(topk_indices_absolute)[0]

            keep_indices = torch.cat([
                torch.arange(start_idx, device=device),
                topk_indices_absolute_sorted
            ]).tolist()

            keep_indices_all_layers.append(keep_indices)

        return keep_indices_all_layers

    @torch.inference_mode()
    def pseudo_forward(self, local_question=None, global_question=None):
        device = self.device

        if local_question is None:
            local_question = "What is happening in the video?"
        if global_question is None:
            global_question = "What is the main topic of the video?"

        local_input_ids = self.processor.tokenizer(local_question).input_ids
        local_input_ids = torch.as_tensor([local_input_ids], device=device, dtype=torch.int)

        global_offset_per_layer = self._get_next_global_offset_per_layer()
        q_len_local = local_input_ids.shape[1]
        batch = local_input_ids.shape[0]

        self._layer_position_ids.clear()
        for layer_idx in range(self.num_layers):
            position_ids_3d = self._build_position_ids_3d_for_text(global_offset_per_layer[layer_idx], q_len_local, batch)
            self._layer_position_ids[layer_idx] = position_ids_3d

        position_ids_local_3d = self._build_position_ids_3d_for_text(global_offset_per_layer[0], q_len_local, batch)

        use_flash_attn = (hasattr(self.language_model.config, '_attn_implementation') and
                        self.language_model.config._attn_implementation in ["flash_attention_2", "sdpa"])

        if use_flash_attn:
            attn_weights_local = self._compute_attention_scores_manually(local_input_ids, self.kv_cache)
        else:
            out_local = self.language_model(
                input_ids=local_input_ids,
                use_cache=False,
                past_key_values=self.kv_cache,
                output_attentions=True,
                position_ids=position_ids_local_3d,
            )
            attn_weights_local = out_local.attentions

        global_input_ids = self.processor.tokenizer(global_question).input_ids
        global_input_ids = torch.as_tensor([global_input_ids], device=device, dtype=torch.int)

        q_len_global = global_input_ids.shape[1]

        self._layer_position_ids.clear()
        for layer_idx in range(self.num_layers):
            position_ids_3d = self._build_position_ids_3d_for_text(global_offset_per_layer[layer_idx], q_len_global, batch)
            self._layer_position_ids[layer_idx] = position_ids_3d

        position_ids_global_3d = self._build_position_ids_3d_for_text(global_offset_per_layer[0], q_len_global, batch)

        if use_flash_attn:
            attn_weights_global = self._compute_attention_scores_manually(global_input_ids, self.kv_cache)
        else:
            out_global = self.language_model(
                input_ids=global_input_ids,
                use_cache=False,
                past_key_values=self.kv_cache,
                output_attentions=True,
                position_ids=position_ids_global_3d,
            )
            attn_weights_global = out_global.attentions

        mixed_question = local_question + "; " + global_question
        mixed_input_ids = self.processor.tokenizer(mixed_question).input_ids
        mixed_input_ids = torch.as_tensor([mixed_input_ids], device=device, dtype=torch.int)

        q_len_mixed = mixed_input_ids.shape[1]

        self._layer_position_ids.clear()
        for layer_idx in range(self.num_layers):
            position_ids_3d = self._build_position_ids_3d_for_text(global_offset_per_layer[layer_idx], q_len_mixed, batch)
            self._layer_position_ids[layer_idx] = position_ids_3d

        position_ids_mixed_3d = self._build_position_ids_3d_for_text(global_offset_per_layer[0], q_len_mixed, batch)

        if use_flash_attn:
            attn_weights_mixed = self._compute_attention_scores_manually(mixed_input_ids, self.kv_cache)
        else:
            out_mixed = self.language_model(
                input_ids=mixed_input_ids,
                use_cache=False,
                past_key_values=self.kv_cache,
                output_attentions=True,
                position_ids=position_ids_mixed_3d,
            )
            attn_weights_mixed = out_mixed.attentions

        self._layer_position_ids.clear()

        print(f"GPU memory usage: {self.get_gpu_memory_usage_gb()} GB")
        current_k_states_len = self.kv_cache[0][0].shape[2]

        keep_indices_all_layers = self.prune_kv_cache_by_attention(
            attn_weights_local, attn_weights_global, attn_weights_mixed,
            num_keep=self.kv_size
        )

        if current_k_states_len > self.kv_size:
            print(f"Applying KV-Cache compression due to k_states > {self.kv_size}")
            self.apply_kv_cache_pruning_strict(keep_indices_all_layers)

    @torch.inference_mode()
    def predict_and_compress(self):
        local_question, global_question = self.predict_next_question()
        self.pseudo_forward(local_question, global_question)

    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=128, temperature=0, repetition_penalty=1.1, pseudo_forward=False):
        device = self.device
        stop_token_ids = [self.processor.tokenizer.eos_token_id]
        output_ids = []

        start_time = time.perf_counter()

        prompt = input_text['prompt']
        input_ids = self.processor.tokenizer(prompt).input_ids
        input_ids = torch.as_tensor([input_ids], device=device)

        self._ensure_dynamic_cache()
        past_lens_prefill = self._get_cache_seq_len_per_layer()
        global_offset_prefill = self._get_next_global_offset_per_layer()

        inputs_embeds = self.get_input_embeddings()(input_ids)
        q_len_prefill = inputs_embeds.shape[1]
        batch = inputs_embeds.shape[0]

        self._layer_position_ids.clear()
        for layer_idx in range(self.num_layers):
            position_ids_3d = self._build_position_ids_3d_for_text(global_offset_prefill[layer_idx], q_len_prefill, batch)
            self._layer_position_ids[layer_idx] = position_ids_3d

        position_ids_3d = self._build_position_ids_3d_for_text(global_offset_prefill[0], q_len_prefill, batch)

        out = self.language_model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=self.kv_cache,
            position_ids=position_ids_3d
        )
        past_key_values = out.past_key_values
        logits = self.lm_head(out.last_hidden_state)

        for layer_idx in range(self.num_layers):
            offset = global_offset_prefill[layer_idx]
            self._append_position_ids_layer(layer_idx, [offset, offset, offset], q_len_prefill)
        self._layer_position_ids.clear()

        for step in range(max_new_tokens):
            last_token_logits = logits[0, -1, :]

            if repetition_penalty != 1.0 and len(output_ids) > 0:
                for token_id in set(output_ids):
                    if last_token_logits[token_id] < 0:
                        last_token_logits[token_id] *= repetition_penalty
                    else:
                        last_token_logits[token_id] /= repetition_penalty

            if temperature == 0.0:
                _, indices = torch.topk(last_token_logits, 1)
                token = int(indices[0])
            else:
                scaled_logits = last_token_logits / temperature
                scaled_logits = torch.nan_to_num(
                    scaled_logits, nan=-float('inf'), posinf=float('inf'), neginf=-float('inf')
                )
                probs = F.softmax(scaled_logits, dim=-1)
                probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
                probs_sum = probs.sum()
                if probs_sum > 0:
                    probs = probs / probs_sum
                    token = torch.multinomial(probs, num_samples=1).item()
                else:
                    _, indices = torch.topk(last_token_logits, 1)
                    token = int(indices[0])

            output_ids.append(token)

            if (not pseudo_forward) and (step == 0):
                end_time = time.perf_counter()
                print(f"TTFT: {end_time - start_time} seconds")

            if token in stop_token_ids:
                break

            curr_global_offset = self._get_next_global_offset_per_layer()

            self._layer_position_ids.clear()
            for layer_idx in range(self.num_layers):
                pos_step_3d = self._build_position_ids_3d_for_text(curr_global_offset[layer_idx], 1, 1)
                self._layer_position_ids[layer_idx] = pos_step_3d

            position_ids_3d = self._build_position_ids_3d_for_text(curr_global_offset[0], 1, 1)

            out = self.language_model(
                input_ids=torch.as_tensor([[token]], device=device),
                use_cache=True,
                past_key_values=past_key_values,
                position_ids=position_ids_3d
            )

            logits = self.lm_head(out.last_hidden_state)
            past_key_values = out.past_key_values

            for layer_idx in range(self.num_layers):
                offset = curr_global_offset[layer_idx]
                self._append_position_ids_layer(layer_idx, [offset, offset, offset], 1)
            self._layer_position_ids.clear()

        output = self.processor.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )

        if not pseudo_forward:
            current_question = input_text['question']
            current_options = None
            formatted_question = input_text.get('formatted_question', None)
            if formatted_question:
                option_matches = re.findall(r'\([A-Z]\)\s*(.+?)(?=\n\([A-Z]\)|\nThe best answer|\n*$)', formatted_question, re.DOTALL)
                if option_matches:
                    current_options = [opt.strip() for opt in option_matches]
            self.conv_history.append((current_question, output, current_options))
            logger.info(f"Saved conversation to history. Total conversations: {len(self.conv_history)}")

        self._truncate_kv_cache(past_lens_prefill)
        for layer_idx in range(self.num_layers):
            if (self._position_ids_cache[layer_idx] is not None and
                self._position_ids_cache[layer_idx].shape[1] > past_lens_prefill[layer_idx]):
                self._position_ids_cache[layer_idx] = self._position_ids_cache[layer_idx][
                    :, :past_lens_prefill[layer_idx]
                ].contiguous()

        new_lens = self._get_cache_seq_len_per_layer()
        print(f"Answering Cache lengths: min={min(new_lens)}, max={max(new_lens)}")
        torch.cuda.empty_cache()
        return output

    def _truncate_kv_cache(self, target_lengths):
        if self.kv_cache is None:
            return

        truncated_cache = []
        for layer_idx, (k_cache, v_cache) in enumerate(self.kv_cache):
            if isinstance(target_lengths, int):
                target_len = target_lengths
            else:
                target_len = target_lengths[layer_idx]

            truncated_k = k_cache[:, :, :target_len, :]
            truncated_v = v_cache[:, :, :target_len, :]
            truncated_cache.append((truncated_k, truncated_v))

        if isinstance(self.kv_cache, DynamicCache):
            self.kv_cache = DynamicCache.from_legacy_cache(truncated_cache)
        else:
            self.kv_cache = truncated_cache


def load_model(model_path='Qwen/Qwen2.5-VL-7B-Instruct',
               n_init=None, kv_size=None, streaming=True, device="cuda", sample_fps=1):
    processor = Qwen2_5_VLProcessor.from_pretrained(model_path)

    system_prompt = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
    init_prompt_ids = processor.tokenizer(system_prompt, return_tensors="pt").input_ids.to(device)

    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    model = QwenVL_Hermes.__new__(QwenVL_Hermes)
    model.__dict__ = base_model.__dict__.copy()

    Abstract_Hermes.__init__(
        model,
        processor,
        init_prompt_ids.tolist(),
        kv_size,
    )
    model.streaming = streaming
    model.sample_fps = sample_fps

    num_layers = base_model.model.config.num_hidden_layers
    model.num_layers = num_layers
    model._position_ids_cache = [None for _ in range(num_layers)]

    model.short_term_ratio = 0.1
    model.long_term_ratio = 0.3
    model.short_term_threshold = int(model.num_layers * model.short_term_ratio)
    model.long_term_threshold = int(model.num_layers * (1 - model.long_term_ratio))

    model.total_processed_frames = 0

    model._mrope_section = _get_mrope_section(base_model.model)

    model._layer_position_ids = {}
    model._hook_handles = []
    model._register_forward_hooks()

    logger.info(f'n_init: {init_prompt_ids.shape[1] if n_init is None else n_init}')
    logger.info(f'kv_size: {kv_size}')

    model.eval()

    return model, processor
