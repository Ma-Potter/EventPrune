"""Evaluate Event Cascade Pruning on ESR-Real."""

import argparse
import json
import logging
import os
import os.path as osp
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import decord
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoProcessor

from qwen_vl_utils import process_vision_info
from qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

decord.bridge.set_bridge("torch")

MCA_QUESTION_TYPES = [
    "rel_distance",
    "rel_dir",
    "proximity",
    "route_planning",
    "obj_sensing",
    "scene_recall",
]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def format_prompt(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    question_type = doc["question_type"]
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") or "These are frames of a video."
    if question_type in MCA_QUESTION_TYPES:
        options = "Options:\n" + "\n".join(doc["options"])
        post_prompt = (
            lmms_eval_specific_kwargs.get("mca_post_prompt", "")
            or "Answer with the option's letter from the given choices directly."
        )
        return "\n".join([pre_prompt, question, options, post_prompt])
    else:
        raise ValueError(
            f"Unknown question type: {question_type}. Expected one of {MCA_QUESTION_TYPES}"
        )


def extract_answer_regex(response, question_type):
    response = response.strip()
    if question_type in MCA_QUESTION_TYPES:
        answer_prefixes = [
            "The best answer is",
            "The correct answer is",
            "The answer is",
            "The answer",
            "Best answer:",
            "Best option:",
            "Option:",
            "Answer:",
            "answer:",
        ]
        for prefix in answer_prefixes:
            response = response.replace(prefix, "")
        response = response.strip()
        standalone_match = re.search("\\b([A-D])\\b", response)
        if standalone_match:
            return standalone_match.group(1).upper()
        matches = re.search("[A-Da-d]", response)
        if matches:
            return matches[0].upper()
        return ""
    else:
        raise ValueError(f"Unexpected question type: {question_type}")


def convert_numpy_types(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    try:
        if hasattr(np, "bool") and isinstance(obj, np.bool):
            return bool(obj)
    except (AttributeError, TypeError):
        pass
    if isinstance(
        obj,
        (
            np.integer,
            np.int_,
            np.intc,
            np.intp,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ),
    ):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def select_frames_by_ets(
    video_path: str,
    event_frame_dir: str,
    target_num_frames: int,
    sampling_fps: int,
    video_id: str,
    threshold: float = 0.1,
) -> list:
    if not os.path.isdir(event_frame_dir):
        logger.error(f"Event frame directory not found for ETS: {event_frame_dir}")
        return None
    try:
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        total_frames = len(vr)
        video_fps = vr.get_avg_fps()
        del vr
    except Exception as e:
        logger.error(f"Could not read video metadata for {video_path}: {e}")
        return None
    if video_fps == 0:
        video_fps = 30
    duration_seconds = total_frames / video_fps
    num_candidate_frames = int(duration_seconds * sampling_fps)
    if num_candidate_frames == 0 or total_frames < 2:
        return list(range(total_frames)) if total_frames > 0 else []
    candidate_indices = (
        torch.linspace(0, total_frames - 1, num_candidate_frames).round().long().tolist()
    )
    event_frames = []
    valid_indices = []

    def load_frame(frame_index):
        frame_pt_path = os.path.join(event_frame_dir, f"{frame_index:010d}.pt")
        if os.path.exists(frame_pt_path):
            try:
                frame_data = torch.load(frame_pt_path, map_location="cpu")
                return (frame_data, frame_index)
            except Exception:
                return (None, frame_index)
        else:
            return (None, frame_index)

    with torch.no_grad():
        with ThreadPoolExecutor(max_workers=16) as executor:
            results = list(executor.map(load_frame, candidate_indices))
    for frame_data, frame_index in results:
        if frame_data is not None:
            event_frames.append(frame_data)
            valid_indices.append(frame_index)
    if len(event_frames) < 2:
        logger.warning(
            f"Video {video_id}: Insufficient event frames ({len(event_frames)}) for ETS. Falling back to all frames."
        )
        return valid_indices if valid_indices else candidate_indices
    if len(event_frames) > 0:
        stacked = torch.stack(event_frames, dim=0)
        frame_scores = stacked.view(stacked.size(0), -1).sum(dim=1).cpu().numpy()
    else:
        frame_scores = np.array([])
    deltas = np.diff(frame_scores)
    if len(deltas) == 0:
        logger.warning(f"Video {video_id}: No deltas computed. Falling back to top-k selection.")
        sorted_indices = np.argsort(frame_scores)[-target_num_frames:]
        return [valid_indices[i] for i in sorted_indices]
    max_delta = np.max(np.abs(deltas))
    if max_delta == 0:
        logger.warning(f"Video {video_id}: All deltas are zero. Falling back to uniform sampling.")
        step = max(1, len(valid_indices) // target_num_frames)
        return valid_indices[::step][:target_num_frames]
    scene_change_threshold = threshold * max_delta
    scene_changes = np.where(np.abs(deltas) > scene_change_threshold)[0]
    keep_frames_set = set()
    for change_idx in scene_changes:
        if change_idx > 0:
            keep_frames_set.add(change_idx - 1)
        keep_frames_set.add(change_idx)
        if change_idx + 1 < len(valid_indices):
            keep_frames_set.add(change_idx + 1)
    activity_threshold = np.quantile(frame_scores, 0.5)
    for t in range(len(frame_scores)):
        if frame_scores[t] > activity_threshold:
            keep_frames_set.add(t)
    keep_frames_absolute = sorted([valid_indices[i] for i in keep_frames_set])
    if len(keep_frames_absolute) == 0:
        step = max(1, len(valid_indices) // target_num_frames) if target_num_frames > 0 else 1
        return valid_indices[::step][:target_num_frames]
    score_map = {valid_indices[i]: frame_scores[i] for i in range(len(valid_indices))}
    all_valid_set = set(valid_indices)
    while len(keep_frames_absolute) > target_num_frames:
        diffs = np.diff(keep_frames_absolute)
        if len(diffs) == 0:
            break
        min_gap_idx = int(np.argmin(diffs))
        frame_a = keep_frames_absolute[min_gap_idx]
        frame_b = keep_frames_absolute[min_gap_idx + 1]
        if score_map.get(frame_a, 0.0) < score_map.get(frame_b, 0.0):
            keep_frames_absolute.pop(min_gap_idx)
        else:
            keep_frames_absolute.pop(min_gap_idx + 1)
    while len(keep_frames_absolute) < target_num_frames:
        if not keep_frames_absolute:
            mid_frame = valid_indices[len(valid_indices) // 2]
            keep_frames_absolute.append(mid_frame)
            keep_frames_absolute.sort()
            continue
        current_set = set(keep_frames_absolute)
        gaps = []
        if keep_frames_absolute[0] > valid_indices[0]:
            gaps.append(
                (
                    valid_indices[0],
                    keep_frames_absolute[0],
                    keep_frames_absolute[0] - valid_indices[0],
                )
            )
        for i in range(len(keep_frames_absolute) - 1):
            start = keep_frames_absolute[i]
            end = keep_frames_absolute[i + 1]
            dist = end - start
            if dist > 1:
                gaps.append((start, end, dist))
        if keep_frames_absolute[-1] < valid_indices[-1]:
            gaps.append(
                (
                    keep_frames_absolute[-1],
                    valid_indices[-1],
                    valid_indices[-1] - keep_frames_absolute[-1],
                )
            )
        if not gaps:
            unselected = [x for x in valid_indices if x not in current_set]
            if not unselected:
                break
            best_remaining = max(unselected, key=lambda x: score_map.get(x, 0.0))
            keep_frames_absolute.append(best_remaining)
            keep_frames_absolute.sort()
            continue
        gaps.sort(key=lambda x: x[2], reverse=True)
        gap_start, gap_end, _ = gaps[0]
        target_mid = (gap_start + gap_end) / 2.0
        candidates = [
            x for x in valid_indices if gap_start <= x <= gap_end and x not in current_set
        ]
        if candidates:
            best_candidate = min(candidates, key=lambda x: abs(x - target_mid))
            keep_frames_absolute.append(best_candidate)
            keep_frames_absolute.sort()
        else:
            unselected = [x for x in valid_indices if x not in current_set]
            if not unselected:
                break
            best_remaining = max(unselected, key=lambda x: score_map.get(x, 0.0))
            keep_frames_absolute.append(best_remaining)
            keep_frames_absolute.sort()
    selected_indices = keep_frames_absolute
    return selected_indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_name", default="ecp_esr_real")
    parser.add_argument("--ckpt_path", default="models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--result_dir", default="results/esr")
    parser.add_argument("--task_jsonl", default="data/ESR-Real/test.jsonl")
    parser.add_argument("--video_dir", default="data/ESR-Real/videos")
    parser.add_argument("--event_voxel_dir", default="data/ESR-Real/event")
    parser.add_argument("--min_pixels", type=int, default=448 * 448)
    parser.add_argument("--max_pixels", type=int, default=448 * 448)
    parser.add_argument("--min_frames", type=int, default=4)
    parser.add_argument("--max_frames", type=int, default=900)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--sampling_mode", choices=["fps", "fixed_frames"], default="fixed_frames")
    parser.add_argument("--num_fixed_frames", type=int, default=64)
    parser.add_argument("--sampling_strategy", choices=["uniform", "ets"], default="ets")
    parser.add_argument("--ets_threshold", type=float, default=0.1)
    parser.add_argument("--emsf_drop_fraction", type=float, default=0.3)
    parser.add_argument("--use_earf", action="store_true")
    config_args = parser.add_mutually_exclusive_group()
    config_args.add_argument("--pruning_config", help="Layer configuration as JSON.")
    config_args.add_argument("--pruning_config_file", help="Path to the layer configuration JSON.")
    args = parser.parse_args()
    if args.num_fixed_frames < 1 or args.fps < 1:
        parser.error("Frame count and sampling FPS must be positive.")
    if not 0 <= args.emsf_drop_fraction <= 1 or not 0 <= args.ets_threshold <= 1:
        parser.error("EMSF fraction and ETS threshold must be in [0, 1].")
    if args.use_earf and not (args.pruning_config or args.pruning_config_file):
        parser.error("EARF requires --pruning_config_file or --pruning_config.")

    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name
    checkpoint_path = args.ckpt_path
    result_dir = args.result_dir
    task_jsonl = args.task_jsonl
    video_dir = args.video_dir
    event_dir = args.event_voxel_dir
    log_path = osp.join(result_dir, "log", f"{run_name}_{curr_time}.log")
    output_jsonl = osp.join(result_dir, "output", f"{run_name}_{curr_time}.jsonl")
    os.makedirs(osp.join(result_dir, "output"), exist_ok=True)
    os.makedirs(osp.join(result_dir, "log"), exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)5s | %(message)s")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    pruning_layers_config = None
    if args.pruning_config_file:
        with open(args.pruning_config_file) as handle:
            pruning_layers_config = json.load(handle)
    elif args.pruning_config:
        pruning_layers_config = json.loads(args.pruning_config)
    logger.info(
        "Sampling: %s; frames: %s; FPS: %s; ETS threshold: %s",
        args.sampling_strategy,
        args.num_fixed_frames,
        args.fps,
        args.ets_threshold,
    )
    logger.info("EMSF drop fraction: %s; EARF: %s", args.emsf_drop_fraction, args.use_earf)
    logger.info("Pruning layers: %s", pruning_layers_config)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        use_emsf=True,
        use_earf=args.use_earf,
        pruning_layers_config=pruning_layers_config,
    )
    processor = AutoProcessor.from_pretrained(
        checkpoint_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    logger.info("Loaded model and processor.")

    task_data = []
    with open(task_jsonl) as handle:
        for line in handle:
            if line.strip():
                task_data.append(json.loads(line))
    task_data = sorted(task_data, key=lambda doc: doc["question_type"])
    logger.info("Loaded %s questions.", len(task_data))

    cnt_total = defaultdict(int)
    cnt_correct = defaultdict(int)
    for doc in tqdm(task_data, total=len(task_data)):
        try:
            video_id = doc["video_id"]
            question_type = doc["question_type"]
            question = doc["question"]
            ground_truth = doc["ground_truth"]
            if question_type not in MCA_QUESTION_TYPES:
                raise ValueError(f"Unsupported question type: {question_type}")
            video_path = osp.join(video_dir, f"{video_id}.mp4")
            if not osp.exists(video_path):
                logger.warning("Video not found; skipping: %s", video_path)
                continue
            event_frame_dir_path = osp.join(event_dir, f"{video_id}_voxel")
            if not osp.exists(event_frame_dir_path):
                event_frame_dir_path = osp.join(event_dir, str(video_id))
            if not osp.exists(event_frame_dir_path):
                logger.warning("Event directory not found; skipping: %s", event_frame_dir_path)
                continue

            selected_frame_indices = None
            if args.sampling_strategy == "ets":
                selected_frame_indices = select_frames_by_ets(
                    video_path=video_path,
                    event_frame_dir=event_frame_dir_path,
                    target_num_frames=args.num_fixed_frames,
                    sampling_fps=args.fps,
                    video_id=video_id,
                    threshold=args.ets_threshold,
                )
                if not selected_frame_indices:
                    logger.warning(
                        "ETS returned no frames for %s; using uniform sampling.", video_id
                    )

            video_params = {
                "type": "video",
                "video": video_path,
                "min_pixels": args.min_pixels,
                "max_pixels": args.max_pixels,
                "event_frame_dir": event_frame_dir_path,
                "merge_size": 2,
            }
            if selected_frame_indices is not None and len(selected_frame_indices) > 0:
                video_params["specific_frame_indices"] = selected_frame_indices
            elif args.sampling_mode == "fps":
                video_params.update(
                    {
                        "fps": args.fps,
                        "min_frames": args.min_frames,
                        "max_frames": args.max_frames,
                    }
                )
            else:
                video_params.update({"nframes": args.num_fixed_frames})

            lmms_eval_specific_kwargs = {
                "pre_prompt": "",
                "mca_post_prompt": "Answer with the option's letter from the given choices directly.",
            }
            text_prompt = format_prompt(doc, lmms_eval_specific_kwargs)
            messages = [
                {
                    "role": "user",
                    "content": [video_params, {"type": "text", "text": text_prompt}],
                }
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs, event_inputs = process_vision_info(
                messages, event_pooling_method="sum"
            )
            if event_inputs is None:
                logger.warning("Event inputs are missing for %s", video_id)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(torch.device("cuda"))
            if event_inputs is not None:
                event_inputs = [event.to(torch.device("cuda")) for event in event_inputs]
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                num_beams=1,
                temperature=None,
                top_p=None,
                top_k=None,
                emsf_drop_fraction=args.emsf_drop_fraction,
                event_inputs=event_inputs,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            response = output_text[0]
            predicted_answer = extract_answer_regex(response, question_type)
            is_correct = predicted_answer.lower() == ground_truth.lower()
            cnt_total["overall"] += 1
            cnt_total[question_type] += 1
            if is_correct:
                cnt_correct["overall"] += 1
                cnt_correct[question_type] += 1
            output_dict = {
                "id": doc.get("id", -1),
                "video_id": video_id,
                "question_type": question_type,
                "question": question,
                "ground_truth": ground_truth,
                "predicted_answer": predicted_answer,
                "response": response,
                "is_correct": is_correct,
                "selected_frame_indices": selected_frame_indices,
            }
            output_dict = convert_numpy_types(output_dict)
            with open(output_jsonl, "a" if osp.exists(output_jsonl) else "w") as handle:
                handle.write(json.dumps(output_dict) + "\n")
        except Exception as error:
            logger.error(
                "Error processing %s: %s", doc.get("video_id", "unknown"), error, exc_info=True
            )

    total = cnt_total["overall"]
    correct = cnt_correct["overall"]
    if total:
        logger.info(
            "Total: %s, Correct: %s, Accuracy: %.1f%%", total, correct, 100 * correct / total
        )
    else:
        logger.info("No question processed.")
    for question_type in MCA_QUESTION_TYPES:
        count = cnt_total[question_type]
        if count:
            logger.info(
                "%s: %s/%s = %.1f%%",
                question_type,
                cnt_correct[question_type],
                count,
                100 * cnt_correct[question_type] / count,
            )
