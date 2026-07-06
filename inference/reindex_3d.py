import torch
from typing import Tuple
from transformers import DynamicCache
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb

def get_cache_seq_len(past_key_values) -> int:
    if past_key_values is None:
        return 0
    if isinstance(past_key_values, DynamicCache):
        return past_key_values.get_seq_length()
    return past_key_values[0][0].shape[2]

def contiguous_kv(past_key_values):
    new_legacy = []
    for (k_layer, v_layer) in past_key_values:
        new_legacy.append((k_layer.contiguous(), v_layer.contiguous()))
    return new_legacy

def _get_rotary_module(llm) -> torch.nn.Module:
    if hasattr(llm, "rotary_emb"):
        return llm.rotary_emb
    if hasattr(llm, "model") and hasattr(llm.language_model, "rotary_emb"):
        return llm.language_model.rotary_emb
    if hasattr(llm, "layers"):
        if len(llm.layers) > 0 and hasattr(llm.layers[0], "self_attn"):
            if hasattr(llm.layers[0].self_attn, "rotary_emb"):
                return llm.layers[0].self_attn.rotary_emb
    if hasattr(llm, "model") and hasattr(llm.language_model, "layers"):
        if len(llm.langauge_model.layers) > 0 and hasattr(llm.language_model.layers[0], "self_attn"):
            if hasattr(llm.language_model.layers[0].self_attn, "rotary_emb"):
                return llm.langauge_model.layers[0].self_attn.rotary_emb
    #raise AttributeError("Cannot find rotary_emb module on language_model")

def _get_mrope_section(llm) -> Tuple[int, int, int]:
    cfg = getattr(llm, "config", None)
    if cfg is None:
        return (16, 24, 24)
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
    
    Args:
        llm: language model
        seq_len: 序列长度
        position_ids_3d: 3D 位置 ID，形状 [3, 1, seq_len] 或 [3, seq_len]
        dtype: 数据类型
        device: 设备
        
    Returns:
        cos, sin: 形状适合 apply_multimodal_rotary_pos_emb
    """
    rotary_emb = _get_rotary_module(llm)
    hidden_size = getattr(llm.config, "hidden_size", None)
    if hidden_size is None and hasattr(llm, "model") and hasattr(llm.model, "config"):
        hidden_size = getattr(llm.model.config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = 4096

    # 确保 position_ids_3d 是 [3, 1, seq_len] 格式
    if position_ids_3d.dim() == 2:
        position_ids_3d = position_ids_3d.unsqueeze(1)  # [3, seq_len] -> [3, 1, seq_len]
    
    pos = position_ids_3d.to(device)
    dummy_h = torch.zeros((1, seq_len, hidden_size), device=device, dtype=dtype)
    cos, sin = rotary_emb(dummy_h, pos)
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
    """
    # 复用多模态 RoPE 接口；它会对 q/k 同时应用，我们丢弃 q 分支
    q_rot, k_rot = apply_multimodal_rotary_pos_emb(
        key_states,  # dummy query
        key_states,
        cos_delta,
        sin_delta,
        mrope_section,
    )
    return k_rot
