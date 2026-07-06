import math
import os
import torch
from logzero import logger
from tqdm import tqdm

from video_qa.base import BaseVQA, work


class HermesVQA(BaseVQA):
    """
    Unified VQA class for both offline and streaming benchmarks.
    
    Streaming mode is auto-detected per conversation sample:
    if a sample has 'end_time', frames are encoded up to that timestamp;
    otherwise all frames are encoded before answering.
    """

    @torch.inference_mode()
    def analyze_a_video(self, video_sample, encode_chunk_size=16):
        video_path = video_sample['video_path']

        video_fps = video_sample.get('fps', None)
        clip = video_sample.get('clip', None)

        if video_path.endswith('.npy'):
            video = self.load_video(video_path, clip=clip)
            video_tensor = torch.from_numpy(video)
        elif os.path.isdir(video_path):
            if video_fps is None:
                raise ValueError(f"video_fps must be provided for image-based video: {video_path}")
            video = self.load_video_frames(video_path, video_fps, clip=clip)
            video_tensor = torch.from_numpy(video)
        else:
            video = self.load_video(video_path, clip=clip)
            video_tensor = torch.from_numpy(video)

        self.qa_model.clear_cache()
        self.qa_model.encode_init_prompt()

        current_frame_idx = 0

        for sample in tqdm(video_sample['conversations']):
            logger.debug(f'sample: {sample}')
            question = sample['question']
            answer = sample['answer']

            if 'end_time' in sample:
                end_frame_idx = math.ceil(sample['end_time'] * self.sample_fps)
            else:
                end_frame_idx = len(video_tensor)

            while current_frame_idx < end_frame_idx:
                next_encode_end = min(current_frame_idx + encode_chunk_size, end_frame_idx)
                if next_encode_end > current_frame_idx:
                    print(f"Encoding frames {current_frame_idx} to {next_encode_end-1}")
                    video_chunk = video_tensor[current_frame_idx:next_encode_end]
                    self.qa_model.encode_video_chunk(video_chunk)
                    current_frame_idx = next_encode_end

                    logger.info(f"Triggering question prediction and KV compression")
                    self.qa_model.predict_and_compress()

            if 'choices' in sample:
                choices = sample['choices']
                if answer is None:
                    answer = choices[0]
                correct_choice = self.choice_letters[choices.index(answer)]
                qa_results = self.video_close_qa(question, choices, correct_choice)
                print("Pred Answer: ", qa_results['pred_answer'])

                record_entry = {
                    'video_id': video_sample['video_id'],
                    'question': question,
                    'choices': choices,
                    'answer': answer,
                    'correct_choice': correct_choice,
                    'pred_answer': qa_results['pred_answer'],
                    'pred_choice': qa_results['pred_choice'],
                    'qa_acc': qa_results['acc'] * 100,
                }
            else:
                qa_results = self.video_open_qa(question, max_new_tokens=256)
                print("Pred Answer: ", qa_results['pred_answer'])

                record_entry = {
                    'video_id': video_sample['video_id'],
                    'question': question,
                    'answer': answer,
                    'pred_answer': qa_results['pred_answer'],
                }

            task = sample.get('task', sample.get('question_type', video_sample.get('task', None)))
            if task is not None:
                record_entry['task'] = task

            duration_category = video_sample.get('duration_category', None)
            if duration_category is not None:
                record_entry['duration_category'] = duration_category

            self.record.append(record_entry)


if __name__ == "__main__":
    work(HermesVQA)
