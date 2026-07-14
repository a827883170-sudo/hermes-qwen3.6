import torch  
from typing import Tuple  
from transformers import DynamicCache  
  
def get_cache_seq_len(past_key_values) -> int:  
    if past_key_values is None:  
        return 0  
    if isinstance(past_key_values, DynamicCache):  
        return past_key_values.get_seq_length()  
    return past_key_values[0][0].shape[2]  
  
def contiguous_kv(past_key_values):  
    if isinstance(past_key_values, DynamicCache):  
        # 对于 DynamicCache，不进行任何操作  
        # transformers 内部已经处理了内存管理  
        return past_key_values  
    else:  
        # Legacy cache 格式  
        new_legacy = []  
        for (k_layer, v_layer) in past_key_values:  
            new_legacy.append((k_layer.contiguous(), v_layer.contiguous()))  
        return new_legacy  
  
def _get_rotary_module(llm) -> torch.nn.Module:  
    # Qwen3.5 特定路径  
    if hasattr(llm, "model") and hasattr(llm.model, "language_model"):  
        if hasattr(llm.model.language_model, "rotary_emb"):  
            return llm.model.language_model.rotary_emb  
      
    # Qwen2.5-VL 路径（向后兼容）  
    if hasattr(llm, "rotary_emb"):  
        return llm.rotary_emb  
    if hasattr(llm, "model") and hasattr(llm.language_model, "rotary_emb"):  
        return llm.language_model.rotary_emb  
    if hasattr(llm, "layers"):  
        if len(llm.layers) > 0 and hasattr(llm.layers[0], "self_attn"):  
            if hasattr(llm.layers[0].self_attn, "rotary_emb"):  
                return llm.layers[0].self_attn.rotary_emb  
    if hasattr(llm, "model") and hasattr(llm.language_model, "layers"):  
        if len(llm.language_model.layers) > 0 and hasattr(llm.language_model.layers[0], "self_attn"):  
            if hasattr(llm.language_model.layers[0].self_attn, "rotary_emb"):  
                return llm.language_model.layers[0].self_attn.rotary_emb  
    #raise AttributeError("Cannot find rotary_emb module on language_model")  
  
def _get_mrope_section(llm) -> Tuple[int, int, int]:  
    cfg = getattr(llm, "config", None)  
    if cfg is None:  
        return (16, 24, 24)  
      
    # Qwen3.5 特定路径：rope_parameters["mrope_section"]  
    if hasattr(cfg, "rope_parameters"):  
        rope_params = cfg.rope_parameters  
        if isinstance(rope_params, dict) and "mrope_section" in rope_params:  
            sec = rope_params["mrope_section"]  
            if isinstance(sec, (list, tuple)) and len(sec) == 3:  
                return tuple(sec)  
      
    # Qwen2.5-VL 路径（向后兼容）  
    text_cfg = getattr(cfg, "text_config", None)  
    if text_cfg and getattr(text_cfg, "rope_scaling", None):  
        sec = text_cfg.rope_scaling.get("mrope_section", None)  
        if isinstance(sec, (list, tuple)) and len(sec) == 3:  
            return tuple(sec)  
    rope_scaling = getattr(cfg, "rope_scaling", None)  
    if rope_scaling and isinstance(rope_scaling, dict) and "mrope_section" in rope_scaling:  
        sec = rope_scaling["mrope_section"]  
        if isinstance(sec, (list, tuple)) and len(sec) == 3:  
            return tuple(sec)  
    return (16, 24, 24)  
  
def compute_cos_sin_for_positions(llm, seq_len: int, position_ids_3d: torch.Tensor, dtype: torch.dtype, device: torch.device):  
    """  
    为给定的 3D 位置计算 cos 和 sin（用于 M-RoPE）  
      
    对于 Qwen3.5，直接使用其原生的 rotary_emb.forward()，该方法内部处理了交错布局  
    """  
    rotary_emb = _get_rotary_module(llm)  
    hidden_size = getattr(llm.config, "hidden_size", None)  
    if hidden_size is None and hasattr(llm, "model") and hasattr(llm.model, "config"):  
        hidden_size = getattr(llm.model.config, "hidden_size", None)  
    if hidden_size is None:  
        hidden_size = 4096  
  
    # 确保 position_ids_3d 是 [3, 1, seq_len] 或 [3, batch, seq_len] 格式  
    if position_ids_3d.dim() == 2:  
        position_ids_3d = position_ids_3d.unsqueeze(1)  # [3, seq_len] -> [3, 1, seq_len]  
      
    # 创建 dummy hidden states，Qwen3.5 的 rotary_emb 需要 hidden_states 作为输入  
    # 但实际上只使用其形状信息  
    if position_ids_3d.shape[1] == 1:  
        batch_size = 1  
    else:  
        batch_size = position_ids_3d.shape[1]  
      
    dummy_h = torch.zeros((batch_size, seq_len, hidden_size), device=device, dtype=dtype)  
      
    # Qwen3.5 的 rotary_emb.forward() 会自动处理交错布局  
    cos, sin = rotary_emb(dummy_h, position_ids_3d)  
    cos = cos.to(dtype)  
    sin = sin.to(dtype)  
    return cos, sin  
  
def rotary_delta(cos_old, sin_old, cos_new, sin_new):  
    # cos(a-b) = cos a cos b + sin a sin b; sin(a-b) = sin a cos b - cos a sin b  
    cos_delta = cos_new * cos_old + sin_new * sin_old  
    sin_delta = sin_new * cos_old - cos_new * sin_old  
    return cos_delta, sin_delta  
  
def apply_rotary_delta_to_keys_only(key_states: torch.Tensor, cos_delta, sin_delta, mrope_section):  
    """  
    对 key states 应用旋转增量（3D M-RoPE 版本）  
      
    注意：对于 Qwen3.5，由于使用了交错布局，直接使用 apply_multimodal_rotary_pos_emb  
    可能不兼容。这里保持原有实现，但在实际使用中可能需要使用 Qwen3.5 原生的方法。  
    """  
    # 复用多模态 RoPE 接口；它会对 q/k 同时应用，我们丢弃 q 分支  
    # 注意：这可能不适用于 Qwen3.5 的交错布局  
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb  
      
    q_rot, k_rot = apply_multimodal_rotary_pos_emb(  
        key_states,  # dummy query  
        key_states,  
        cos_delta,  
        sin_delta,  
        mrope_section,  
    )  
    return k_rot
