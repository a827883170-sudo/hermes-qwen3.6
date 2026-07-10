import torch  
import itertools  
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration, DynamicCache  
from inference.abstract_hermes import Abstract_Hermes  
from inference.reindex_3d import (  
    get_cache_seq_len,  
    contiguous_kv,  
    _get_mrope_section,  
)  
import re
import torch.nn.functional as F  
import transformers.modeling_flash_attention_utils  
  
# Flash Attention 补丁  
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
  
  
class Qwen3_5VL_Hermes(Qwen3_5ForConditionalGeneration, Abstract_Hermes):  
    """Qwen3.5-VL with HERMES strict-shrink support."""  
  
    def __init__(self, config, processor, init_prompt_ids, kv_size, streaming=True, sample_fps=1):  
        Abstract_Hermes.__init__(self, processor, init_prompt_ids, kv_size)  
        self.streaming = streaming  
        self.sample_fps = sample_fps  
  
        num_layers = config.text_config.num_hidden_layers if hasattr(config, 'text_config') else 28  
        self.num_layers = num_layers  
  
        self.short_term_ratio = 0.3  
        self.long_term_ratio = 0.3  
        self.short_term_threshold = int(self.num_layers * self.short_term_ratio)  
        self.long_term_threshold = int(self.num_layers * (1 - self.long_term_ratio))  
  
        self._position_ids_cache = [None for _ in range(num_layers)]  
        self._layer_position_ids = {}  
        self._hook_handles = []  
        self.total_processed_frames = 0  
  
        self._mrope_section = _get_mrope_section(self.model.language_model)  
        self._register_forward_hooks()  
  
    def _ensure_dynamic_cache(self):  
        if self.kv_cache is None:  
            return  
        if not isinstance(self.kv_cache, DynamicCache):  
            self.kv_cache = DynamicCache.from_legacy_cache(self.kv_cache)  
  
    def _register_forward_hooks(self):  
        def make_hook(layer_idx):  
            def hook(module, args, kwargs):  
                if layer_idx in self._layer_position_ids:  
                    kwargs['position_ids'] = self._layer_position_ids[layer_idx]  
                return args, kwargs  
            return hook  
  
        for layer_idx, layer in enumerate(self.model.language_model.layers):  
            handle = layer.register_forward_pre_hook(make_hook(layer_idx), with_kwargs=True)  
            self._hook_handles.append(handle)  
  
    def _clear_forward_hooks(self):  
        for handle in self._hook_handles:  
            handle.remove()  
        self._hook_handles = []  
  
    def _get_cache_seq_len_per_layer(self) -> list:  
        if self.kv_cache is None:  
            return [0] * self.num_layers  
  
        lengths = []  
        if isinstance(self.kv_cache, DynamicCache):  
            for layer_idx in range(self.num_layers):  
                lengths.append(self.kv_cache.get_seq_length())  
        else:  
            for layer_idx in range(len(self.kv_cache)):  
                k_layer, v_layer = self.kv_cache[layer_idx]  
                lengths.append(k_layer.shape[2])  
        return lengths  
  
    def _get_next_global_offset_per_layer(self) -> list:  
        if self.kv_cache is None:  
            return [0] * self.num_layers  
          
        offsets = []  
        if isinstance(self.kv_cache, DynamicCache):  
            for layer_idx in range(self.num_layers):  
                offsets.append(self.kv_cache.get_seq_length())  
        else:  
            for layer_idx in range(len(self.kv_cache)):  
                k_layer, v_layer = self.kv_cache[layer_idx]  
                offsets.append(k_layer.shape[2])  
        return offsets  
  
    def _build_position_ids_3d_for_vision(self, grid_pos_ids: torch.Tensor, batch: int) -> torch.Tensor:  
        """[3, batch, q_len] -- vision tokens use grid-based 3D positions."""  
        # 确保 grid_pos_ids 是 (3, seq_len) 形状  
        if grid_pos_ids.dim() == 3 and grid_pos_ids.shape[0] == 1:  
            grid_pos_ids = grid_pos_ids.squeeze(0)  # [1, 3, seq_len] -> [3, seq_len]  
        
        # 参考 qwenvl_hermes.py 的实现  
        pos_3d = grid_pos_ids.unsqueeze(1).expand(3, batch, -1).clone()  
        return pos_3d
  
    def _append_position_ids_layer_explicit(self, layer_idx: int, new_pos: torch.Tensor):  
        # 处理可能的额外 batch 维度  
        if new_pos.dim() == 3 and new_pos.shape[0] == 1:  
            new_pos = new_pos.squeeze(0)  # [1, 3, seq_len] -> [3, seq_len]  
        
        if self._position_ids_cache[layer_idx] is None:  
            self._position_ids_cache[layer_idx] = new_pos.to(self.device)  
        else:  
            self._position_ids_cache[layer_idx] = torch.cat(  
                [self._position_ids_cache[layer_idx], new_pos.to(self.device)], dim=1  
            )
  
    @torch.inference_mode()  
    def encode_init_prompt(self):  
        if not isinstance(self.init_prompt_ids, torch.Tensor):  
            self.init_prompt_ids = torch.as_tensor(self.init_prompt_ids, device=self.device)  
  
        seq_len = self.init_prompt_ids.shape[-1]  
        pos_1d = torch.arange(seq_len, device=self.device, dtype=torch.float32)  
        position_ids_3d = pos_1d.unsqueeze(0).unsqueeze(0).expand(3, 1, -1).clone()  
  
        output = self.model.language_model(  
            input_ids=self.init_prompt_ids,  
            use_cache=True,  
            return_dict=True,  
            position_ids=position_ids_3d,  
        )  
        self.kv_cache = output.past_key_values  
          
        # 使用 get_cache_seq_len 获取序列长度  
        self.visual_start_idx = get_cache_seq_len(self.kv_cache)  
  
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
    
        # 使用 processor 获取视频数据  
        video_input = self.processor(text=[""], videos=video_chunk, return_tensors="pt").to(self.device, self.dtype)  
        pixel_values_videos = video_input["pixel_values_videos"]  
        video_grid_thw = video_input["video_grid_thw"]  
        
        # 获取当前缓存长度作为起始位置  
        current_seq_len = get_cache_seq_len(self.kv_cache)  
        
        # 使用原生方法生成位置 ID  
        grid_pos_ids = self.model.get_vision_position_ids(  
            start_position=current_seq_len,  
            grid_thw=video_grid_thw[0],  
            spatial_merge_size=self.model.config.vision_config.spatial_merge_size,  
            device=self.device  
        )  
        
        # 获取视觉特征（已通过 merger 投影）  
        video_outputs = self.get_video_features(pixel_values_videos, video_grid_thw)  
        video_embeds = video_outputs.pooler_output  
        video_embeds = torch.cat(video_embeds, dim=0).unsqueeze(0)  
        
        # 构建 3D position IDs  
        default_position_ids_3d = self._build_position_ids_3d_for_vision(grid_pos_ids, batch=1)  
        
        # 获取每层的全局偏移量  
        global_offset_per_layer = self._get_next_global_offset_per_layer()  
        base_offset = global_offset_per_layer[0]  
        
        # 直接传递给 language_model  
        out = self.model.language_model(  
            inputs_embeds=video_embeds,  
            past_key_values=self.kv_cache,  
            use_cache=True,  
            return_dict=True,  
            position_ids=default_position_ids_3d,  
        )  
        self.kv_cache = out.past_key_values  
        contiguous_kv(self.kv_cache)  
        
        # 更新位置 IDs 缓存  
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
  
    def get_prompt(self, query, mc=False):  
        """生成提示词，与 Abstract_Hermes 基类保持一致"""  
        prompt = f"\n{query}<|im_end|><|im_start|>assistant\n"  
        
        if mc:  
            prompt += 'Best option: ('  
        return prompt 
    
    def predict_next_question(self):  
        """预测下一个问题，用于 KV cache 压缩"""  
        # 简单实现：返回固定的本地和全局问题  
        local_question = "What is happening in the video?"  
        global_question = "What is the main topic of the video?"  
        return local_question, global_question  
    
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
    
        # 暂时禁用 Flash Attention，使用 output_attentions=True  
        out_local = self.model.language_model(  
            input_ids=local_input_ids,  
            use_cache=False,  
            past_key_values=self.kv_cache,  
            output_attentions=True,  
            position_ids=position_ids_local_3d,  
        )  
        attn_weights_local = out_local.attentions  
    
        # 对 global 和 mixed 问题做同样处理  
        global_input_ids = self.processor.tokenizer(global_question).input_ids  
        global_input_ids = torch.as_tensor([global_input_ids], device=device, dtype=torch.int)  
    
        q_len_global = global_input_ids.shape[1]  
    
        self._layer_position_ids.clear()  
        for layer_idx in range(self.num_layers):  
            position_ids_3d = self._build_position_ids_3d_for_text(global_offset_per_layer[layer_idx], q_len_global, batch)  
            self._layer_position_ids[layer_idx] = position_ids_3d  
    
        position_ids_global_3d = self._build_position_ids_3d_for_text(global_offset_per_layer[0], q_len_global, batch)  
    
        out_global = self.model.language_model(  
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
    
        out_mixed = self.model.language_model(  
            input_ids=mixed_input_ids,  
            use_cache=False,  
            past_key_values=self.kv_cache,  
            output_attentions=True,  
            position_ids=position_ids_mixed_3d,  
        )  
        attn_weights_mixed = out_mixed.attentions  
    
        self._layer_position_ids.clear()  
    
        print(f"GPU memory usage: {self.get_gpu_memory_usage_gb()} GB")  
        current_k_states_len = get_cache_seq_len(self.kv_cache)  
    
        keep_indices_all_layers = self.prune_kv_cache_by_attention(  
            attn_weights_local, attn_weights_global, attn_weights_mixed,  
            num_keep=self.kv_size  
        )  
    
        if current_k_states_len > self.kv_size:  
            print(f"Applying KV-Cache compression due to k_states > {self.kv_size}")  
            self.apply_kv_cache_pruning_strict(keep_indices_all_layers)
    
    @torch.inference_mode()  
    def predict_and_compress(self):  
        """预测下一个问题并进行 KV cache 压缩"""  
        local_question, global_question = self.predict_next_question()  
        self.pseudo_forward(local_question, global_question)

    def _build_position_ids_3d_for_text(self, global_offset: int, q_len: int, batch: int) -> torch.Tensor:  
        """[3, batch, q_len] -- text tokens use identical position across all 3 dims."""  
        pos_1d = torch.arange(global_offset, global_offset + q_len, device=self.device, dtype=torch.float32)  
        pos_2d = pos_1d.unsqueeze(0).expand(batch, -1)  
        pos_3d = pos_2d.unsqueeze(0).expand(3, -1, -1).clone()  
        return pos_3d

    def allocate_budget_by_depth(self, total_budget, num_layers):  
        budget_per_layer = [total_budget // num_layers] * num_layers  
        diff = total_budget - sum(budget_per_layer)  
        budget_per_layer[-1] += diff  
        return budget_per_layer  
  
    def prune_kv_cache_by_attention(self, attn_weights_local, attn_weights_global, attn_weights_mixed, num_keep=3000):  
        device = self.device  
        visual_start_idx = self.visual_start_idx  
        num_layers = len(attn_weights_local)  
        layer_types = self.model.language_model.config.layer_types
    
        # 1. 动态获取有效的问题长度（跳过线性层）  
        def get_valid_q_len(weights_list):  
            for w in weights_list:  
                if w is not None and len(w) > 0:  
                    return w[0].shape[2]  
            return 0  
    
        question_len_local = get_valid_q_len(attn_weights_local)  
        question_len_global = get_valid_q_len(attn_weights_global)  
        question_len_mixed = get_valid_q_len(attn_weights_mixed)  
    
        total_budget = num_keep * num_layers  
        budget_per_layer = self.allocate_budget_by_depth(total_budget, num_layers)  
    
        layer_raw_scores = []  
        layer_configs = []  
        full_attn_layer_indices = []  
    
        # 2. 第一遍循环：计算全注意力层的分数  
        for layer_idx in range(num_layers):  
            if layer_types[layer_idx] == "linear_attention":  
                continue  
            
            full_attn_layer_indices.append(layer_idx)  
            
            if layer_idx < self.short_term_threshold:  
                layer_type, layer_attn_weights, q_len, alpha, k = "short-term", attn_weights_local[layer_idx], question_len_local, 1, 20  
            elif layer_idx >= self.long_term_threshold:  
                layer_type, layer_attn_weights, q_len, alpha, k = "long-term", attn_weights_global[layer_idx], question_len_global, 0, 0.0  
            else:  
                layer_type, layer_attn_weights, q_len = "mid-term", attn_weights_mixed[layer_idx], question_len_mixed  
                progress = (layer_idx - self.short_term_threshold) / (self.long_term_threshold - self.short_term_threshold)  
                alpha, k = 0.75 - 0.6 * progress, 20 - 12 * progress  
    
            # 计算注意力分数  
            visual_attn_weights = layer_attn_weights[0].mean(dim=0)[:, visual_start_idx:-1*q_len].mean(dim=0)  
            num_visual_tokens = visual_attn_weights.shape[0]  
            
            positions = torch.arange(num_visual_tokens, device=device, dtype=torch.float32)  
            recency_weights = torch.exp(-k * (num_visual_tokens - 1 - positions) / max(num_visual_tokens - 1, 1))  
    
            attn_norm = (visual_attn_weights - visual_attn_weights.min()) / (visual_attn_weights.max() - visual_attn_weights.min() + 1e-6)  
            recency_norm = (recency_weights - recency_weights.min()) / (recency_weights.max() - recency_weights.min() + 1e-6)  
    
            layer_raw_scores.append(attn_norm * (1 - alpha) + recency_norm * alpha)  
            layer_configs.append({  
                'budget': min(budget_per_layer[layer_idx], num_visual_tokens),  
                'layer_type': layer_type,  
                'visual_start_idx': visual_start_idx  
            })  
    
        # 3. 分数平滑与细化（仅针对全注意力层）  
        refined_scores = [s.clone() for s in layer_raw_scores]  
        for i in range(len(refined_scores) - 2, -1, -1):  
            gamma = 0.4 if layer_configs[i]['layer_type'] == 'long-term' else (0.3 if layer_configs[i]['layer_type'] == 'mid-term' else 0.1)  
            score_next = refined_scores[i+1]  
            if refined_scores[i].shape[0] != score_next.shape[0]:  
                score_next = F.interpolate(score_next.view(1, 1, -1), size=refined_scores[i].shape[0], mode='linear', align_corners=False).view(-1)  
            refined_scores[i] = (1 - gamma) * refined_scores[i] + gamma * score_next  
    
        # 4. 生成最终的 keep_indices 列表  
        keep_indices_all_layers = [None] * num_layers  
        full_attn_ptr = 0  
        for layer_idx in range(num_layers):  
            if layer_types[layer_idx] == "linear_attention":  
                # 线性层：保留所有 token（状态机制不需要裁剪）  
                curr_len = get_cache_seq_len(self.kv_cache) 
                keep_indices_all_layers[layer_idx] = list(range(curr_len))  
            else:  
                # 全注意力层：执行 HERMES 裁剪  
                score = refined_scores[full_attn_ptr]  
                config = layer_configs[full_attn_ptr]  
                topk_idx = torch.topk(score, config['budget'], sorted=False)[1]  
                abs_idx = torch.sort(topk_idx + config['visual_start_idx'])[0]  
                keep_indices_all_layers[layer_idx] = torch.cat([torch.arange(config['visual_start_idx'], device=device), abs_idx]).tolist()  
                full_attn_ptr += 1  
    
        return keep_indices_all_layers

    @torch.inference_mode()  
    def apply_kv_cache_pruning_strict(self, keep_indices_all_layers):  
        if self.kv_cache is None:  
            logger.warning("No KV-Cache to prune")  
            return  
        if not keep_indices_all_layers or len(keep_indices_all_layers[0]) == 0:  
            logger.warning("Empty keep_indices; skip pruning")  
            return  
    
        layer_types = self.model.language_model.config.layer_types  
        
        # 只对全注意力层应用裁剪  
        for layer_idx, keep_indices in enumerate(keep_indices_all_layers):  
            if layer_types[layer_idx] == "linear_attention":  
                # 线性注意力层跳过裁剪  
                continue  
    
        self._shrink_positions_and_rerotate_keys(keep_indices_all_layers)  
        logger.info(f"Strict-shrunk KV cache. New length: {get_cache_seq_len(self.kv_cache)}")  
    
    def _shrink_positions_and_rerotate_keys(self, keep_indices_per_layer):  
        """收缩位置 ID 并重新旋转 keys"""  
        from inference.reindex_3d import (  
            get_cache_seq_len,  
            compute_cos_sin_for_positions,  
            rotary_delta,  
            apply_rotary_delta_to_keys_only,  
        )  
        
        device = self.device  
        dtype = self.dtype  
        should_compact = True  
        mrope_section = self._mrope_section  
        
        old_position_ids_cache = [pos.clone() if pos is not None else None   
                                for pos in self._position_ids_cache]  
        
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
                    self.model.language_model, len(safe_idx), old_pos_kept, dtype, device  
                )  
                cos_new, sin_new = compute_cos_sin_for_positions(  
                    self.model.language_model, len(safe_idx), new_pos_kept, dtype, device  
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
                        self.model.language_model, old_pos_pruned.shape[1], old_pos_pruned, dtype, device  
                    )  
                    cos_new, sin_new = compute_cos_sin_for_positions(  
                        self.model.language_model, target_pos_pruned.shape[1], target_pos_pruned, dtype, device  
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
    
    def _compute_attention_scores_manually(self, input_ids, past_key_values):  
        device = self.device  
        global_offset_per_layer = self._get_next_global_offset_per_layer()  
        q_len = input_ids.shape[1]  
        batch = input_ids.shape[0]  
        
        config = self.model.language_model.config  
        num_layers = config.num_hidden_layers  
        layer_types = config.layer_types  
        
        attention_weights_list = []  
        
        for layer_idx in range(num_layers):  
            # 跳过线性注意力层，返回占位符权重  
            if layer_types[layer_idx] == "linear_attention":  
                # 返回全 1 的占位符权重，表示所有 token 都重要  
                current_len = get_cache_seq_len(past_key_values)  
                placeholder_weights = torch.ones(1, 1, q_len, current_len, device=device)  
                attention_weights_list.append(placeholder_weights)  
                continue  
                
            # 处理全注意力层（原有逻辑）  
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
    
            rotary_emb = _get_rotary_module(self.model.language_model)  
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

    @torch.inference_mode()  
    def question_answering(self, input_text, max_new_tokens=128, temperature=0, repetition_penalty=1.1, pseudo_forward=False):  
        device = self.device  
        stop_token_ids = [self.processor.tokenizer.eos_token_id]  
        output_ids = []  
    
        prompt = input_text['prompt']  
        input_ids = self.processor.tokenizer(prompt).input_ids  
        input_ids = torch.as_tensor([input_ids], device=device)  
    
        self._ensure_dynamic_cache()  
        past_lens_prefill = self._get_cache_seq_len_per_layer()  
        global_offset_prefill = self._get_next_global_offset_per_layer()  
    
        inputs_embeds = self.model.language_model.get_input_embeddings()(input_ids)  
        q_len_prefill = inputs_embeds.shape[1]  
        batch = inputs_embeds.shape[0]  
    
        self._layer_position_ids.clear()  
        for layer_idx in range(self.num_layers):  
            position_ids_3d = self._build_position_ids_3d_for_text(global_offset_prefill[layer_idx], q_len_prefill, batch)  
            self._layer_position_ids[layer_idx] = position_ids_3d  
    
        position_ids_3d = self._build_position_ids_3d_for_text(global_offset_prefill[0], q_len_prefill, batch)  
    
        out = self.model.language_model(  
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
    
            if token in stop_token_ids:  
                break  
    
            curr_global_offset = self._get_next_global_offset_per_layer()  
    
            self._layer_position_ids.clear()  
            for layer_idx in range(self.num_layers):  
                pos_step_3d = self._build_position_ids_3d_for_text(curr_global_offset[layer_idx], 1, 1)  
                self._layer_position_ids[layer_idx] = pos_step_3d  
    
            position_ids_3d = self._build_position_ids_3d_for_text(curr_global_offset[0], 1, 1)  
    
            out = self.model.language_model(  
                input_ids=torch.as_tensor([[token]], device=device),  
                use_cache=True,  
                past_key_values=past_key_values,  
                position_ids=position_ids_3d  
            )  
    
            logits = self.model.lm_head(out.last_hidden_state)  
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
  
  
def load_model(model_path='models/Qwen3.5-VL-7B-Instruct',  
               n_init=None, kv_size=None, streaming=True, device="cuda", sample_fps=1):  
    """Load Qwen3.5 model with HERMES adaptations."""  
      
    # Load processor using AutoProcessor  
    processor = AutoProcessor.from_pretrained(model_path)  
      
    # System prompt for Qwen3.5  
    system_prompt = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'  
    init_prompt_ids = processor.tokenizer(system_prompt, return_tensors="pt").input_ids.tolist()  
      
    # Load base model  
    base_model = Qwen3_5ForConditionalGeneration.from_pretrained(  
        model_path,  
        device_map="auto",  
        torch_dtype=torch.float16,  
        trust_remote_code=True  
    )  
      
    # Create HERMES wrapper  
    model = Qwen3_5VL_Hermes.__new__(Qwen3_5VL_Hermes)  
    model.__dict__ = base_model.__dict__.copy()  
      
    # Initialize HERMES components  
    Qwen3_5VL_Hermes.__init__(  
        model,  
        base_model.config,  
        processor,  
        init_prompt_ids,  
        kv_size,  
        streaming,  
        sample_fps  
    )  
      
    # Set to evaluation mode  
    model.eval()  
      
    return model, processor
