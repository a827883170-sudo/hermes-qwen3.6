import torch

class Abstract_Hermes:
    kv_cache = None

    def __init__(self, processor, init_prompt_ids, kv_size):
        self.processor = processor
        self.init_prompt_ids = init_prompt_ids
        self.kv_size = kv_size
        self.last_encoded_frames = 0
        self.visual_start_idx = 14
        self.conv_history = []
        
    def clear_cache(self):
        self.kv_cache = None
        self.conv_history = []
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def encode_init_prompt(self):
        if not isinstance(self.init_prompt_ids, torch.Tensor):
            self.init_prompt_ids = torch.as_tensor(self.init_prompt_ids, device=self.device)
        output = self.language_model(input_ids=self.init_prompt_ids, use_cache=True, return_dict=True)
        self.kv_cache = output.past_key_values
        self.visual_start_idx = self.kv_cache[0][0].shape[2]

    def get_prompt(self, query, mc=False):
        prompt = f"\n{query}<|im_end|><|im_start|>assistant\n"
        
        if mc:
            prompt += 'Best option: ('
        return prompt

    def get_gpu_memory_usage_gb(self):
        return torch.cuda.max_memory_allocated() / (1024**3)