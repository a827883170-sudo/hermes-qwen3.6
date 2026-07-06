export PYTHONPATH=$(cd "$(dirname "$0")/.." && pwd):$PYTHONPATH
# The number of processes utilized for parallel evaluation.
# Normally, set it to the number of GPUs on your machine.
num_chunks=8

# Supported model: [llava_ov_0.5b, llava_ov_7b, llava_ov_72b, qwen2.5_vl_3b, qwen2.5_vl_7b, qwen2.5_vl_32b]
model=llava_ov_7b

# Supported dataset: [videomme, mvbench, egoschema, rvs_ego, rvs_movie, ovobench, streamingbench]
dataset=streamingbench


python video_qa/run_infer.py \
    --num_chunks $num_chunks \
    --model ${model} \
    --dataset ${dataset} \
    --sample_fps 0.5 \
    --kv_size 6000