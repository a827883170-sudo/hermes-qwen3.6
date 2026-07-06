import csv
import warnings
import random
import json
import os
import math
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image
from decord import VideoReader
from transformers import (
    logging,
)

import logzero
from logzero import logger

from inference.llavaov_hermes import load_model as llavaov_hermes_load_model


def qwenvl_hermes_load_model(*args, **kwargs):
    try:
        from inference.qwenvl_hermes import load_model as _load_model
    except Exception as exc:
        raise ImportError(
            "Failed to import inference.qwenvl_hermes. "
            "Qwen models require a newer transformers version. "
            "Please use llava models on old transformers, or upgrade for qwen."
        ) from exc
    return _load_model(*args, **kwargs)

MODELS = {
    'llava_ov_0.5b': {
        'load_func': llavaov_hermes_load_model,
        'model_path': 'models/llava-onevision-qwen2-0.5b-ov-hf',
    },
    'llava_ov_7b': {
        'load_func': llavaov_hermes_load_model,
        'model_path': 'models/llava-onevision-qwen2-7b-ov-hf',
    },
    'llava_ov_72b': {
        'load_func': llavaov_hermes_load_model,
        'model_path': 'models/llava-onevision-qwen2-72b-ov-hf',
    },
    'qwen2.5_vl_3b': {
        'load_func': qwenvl_hermes_load_model,
        'model_path': 'models/Qwen2.5-VL-3B-Instruct',
    },
    'qwen2.5_vl_7b': {
        'load_func': qwenvl_hermes_load_model,
        'model_path': 'models/Qwen2.5-VL-7B-Instruct',
    },
    'qwen2.5_vl_32b': {
        'load_func': qwenvl_hermes_load_model,
        'model_path': 'models/Qwen2.5-VL-32B-Instruct',
    },
}

class BaseVQA:
    def __init__(self, anno, save_dir, sample_fps,
                 qa_model, qa_processor=None,
                 num_chunks=None, chunk_idx=None) -> None:
        self.sample_fps = sample_fps
        self.qa_model = qa_model
        self.qa_processor = qa_processor

        self.num_chunks = num_chunks
        self.chunk_idx = chunk_idx
        if num_chunks is not None:
            anno = self.get_chunk(anno, num_chunks, chunk_idx)
        self.anno = anno

        self.save_dir = save_dir
        self.choice_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        self.record = []

    def split_list(self, lst, n):
        """Split a list into n (roughly) equal-sized chunks"""
        chunk_size = math.ceil(len(lst) / n)
        return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

    def get_chunk(self, lst, n, k):
        chunks = self.split_list(lst, n)
        return chunks[k]
    
    def load_video(self, video_path, clip=None):
        """
        Load video from file.
        
        Args:
            video_path: Path to the video file (.npy or regular video)
            clip: Optional [start_time, end_time] in seconds to extract a specific segment
            
        Returns:
            For .npy files: numpy array of frames
            For regular videos: (numpy array of frames, resized_height, resized_width)
        """
        if video_path.endswith('.npy'):
            video = np.load(video_path)
            num_frames = len(video)
            frame_idx = np.linspace(0, num_frames-1, int(num_frames*self.sample_fps), dtype=int).tolist()
            video = video[frame_idx]
            return video
        else:
            vr = VideoReader(video_path, num_threads=1)
            fps = round(vr.get_avg_fps())
            total_frames = len(vr)
            
            if clip is not None:
                # Calculate frame range based on clip times
                start_frame = max(0, int(clip[0] * fps))
                end_frame = min(total_frames, int(clip[1] * fps) + 1)
                print(f"start_frame: {start_frame}")
                print(f"end_frame: {end_frame}")
            else:
                start_frame = 0
                end_frame = total_frames
            
            # Sample frames at target fps within the clip range
            sample_step = int(fps / self.sample_fps)
            frame_idx = [i for i in range(start_frame, end_frame, sample_step)]
            video = vr.get_batch(frame_idx).asnumpy()
            return video
    
    def load_video_frames(self, video_path, video_fps, clip=None):
        """
        Load video from a directory of image frames (for OVBench image-based videos).
        
        Args:
            video_path: Path to the directory containing image frames
            video_fps: Original FPS of the video (from annotation)
            clip: Optional [start_time, end_time] to extract a specific segment
            
        Returns:
            video: numpy array of frames
        """
        # Get sorted list of image files
        img_files = sorted(os.listdir(video_path))
        # Filter only image files
        img_files = [f for f in img_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        num_frames = len(img_files)
        
        # Calculate frame indices based on clip times
        if clip is not None:
            start_time, end_time = clip
            start_frame = max(0, int(start_time * video_fps))
            end_frame = min(num_frames - 1, int(end_time * video_fps))
            print(f"start_frame: {start_frame}")
            print(f"end_frame: {end_frame}")
        else:
            start_frame = 0
            end_frame = num_frames - 1
        
        # Generate sampled frame indices based on sample_fps
        sample_step = max(1, int(video_fps / self.sample_fps))
        frame_idx = list(range(start_frame, end_frame + 1, sample_step))
        
        # Load images
        frames = []
        for i in frame_idx:
            if i < len(img_files):
                img_path = os.path.join(video_path, img_files[i])
                img = Image.open(img_path).convert('RGB')
                frames.append(np.array(img))
        
        video = np.stack(frames, axis=0)
        return video
    
    def format_mcqa_prompt(self, question, candidates):
        assert len(question) > 0, f"Q: {question}"

        formatted_choices = "\n".join(["(" + self.choice_letters[i] + ") " + candidate for i, candidate in enumerate(candidates)])
        formatted_question = f"Question: {question}\nOptions:\n{formatted_choices}\nOnly give the best option."

        return {
            "question": f"{question}",
            "formatted_question": formatted_question,
            "prompt": self.qa_model.get_prompt(formatted_question, mc=True)
        }

    def extract_characters_regex(self, s):
        s = s.strip()
        if ")" in s:
            index = s.index(")")
            pred = s[index - 1 : index]
            return pred
        else:
            try:
                return s[0]
            except:
                return s

    def video_open_qa(self, question, max_new_tokens=1024, retrieved_indices=None):
        input_text = {
            "question": question,
            "prompt": self.qa_model.get_prompt(question)
        }
        pred_answer = self.qa_model.question_answering(input_text, max_new_tokens=max_new_tokens)
        return {
            'pred_answer': pred_answer.replace('\n', ''),
        }

    def video_close_qa(self, question, candidates, correct_choice, retrieved_indices=None):
        input_text = self.format_mcqa_prompt(question, candidates)
        pred_answer = self.qa_model.question_answering(input_text, max_new_tokens=16)
        pred_letter = self.extract_characters_regex(pred_answer)
        return {
            'pred_answer': pred_answer.replace('\n', ''),
            'pred_choice': pred_letter,
            'acc': float(pred_letter == correct_choice),
        }

    def pseudo_qa(self, prediction_prompt=None):
        if prediction_prompt is None:
            prediction_prompt = "<|im_end|><|im_start|>assistant\n"
        input_text = {
            "question": prediction_prompt,
            "prompt": self.qa_model.get_prompt(prediction_prompt)
        }
        self.qa_model.pseudo_forward(input_text)

    @torch.inference_mode()
    def analyze_a_video(self, video_sample):
        pass

    def analyze(self, debug=False):
        video_annos = self.anno[:1] if debug else self.anno
        for video_sample in tqdm(video_annos):
            logger.debug(f'video_id: {video_sample["video_id"]}')
            self.analyze_a_video(video_sample)

        final_df = pd.DataFrame(self.record)
        final_df.to_csv(f'{self.save_dir}/{self.num_chunks}_{self.chunk_idx}.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def work(QA_CLASS):
    logging.set_verbosity_error()

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_fps", type=float, default=1)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--anno_path", type=str, required=True)
    parser.add_argument("--model", type=str, default="llava_ov_7b")
    parser.add_argument("--debug", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--kv_size", type=int)
    parser.add_argument("--streaming", type=str2bool, nargs='?', const=True, default=False,
                        help="Streaming (online) mode. If False (default), uses offline mode where should_compact is always True.")
    args = parser.parse_args()

    if not args.debug:
        logzero.loglevel(logging.INFO)
        warnings.filterwarnings('ignore')

    os.makedirs(args.save_dir, exist_ok=True)

    # fix random seed
    random.seed(2024)
    logger.info('seed: 2024')

    # VideoQA model
    model_path = MODELS[args.model]['model_path']
    load_func = MODELS[args.model]['load_func']
    logger.info(f"Loading VideoQA model: {model_path}")
    videoqa_model, videoqa_processor = load_func(
        model_path=model_path,
        kv_size=args.kv_size,
        streaming=args.streaming,
        sample_fps=args.sample_fps,
    )

    # Load ground truth file
    anno = json.load(open(args.anno_path))

    analyzer = QA_CLASS(
        anno=anno,
        sample_fps=args.sample_fps,
        qa_model=videoqa_model,
        qa_processor=videoqa_processor,
        num_chunks=args.num_chunks,
        chunk_idx=args.chunk_idx,
        save_dir=args.save_dir,
    )

    analyzer.analyze(debug=args.debug)
