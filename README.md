# EventPrune

**Cascaded Event-Assisted Token Pruning for Efficient Embodied Spatial Reasoning · NeurIPS 2026**

[Paper](https://arxiv.org/abs/2605.19506) · [Project](https://pengtaoma00.github.io/eventprune/) · [ESR-Real](https://huggingface.co/datasets/pengtaoma00/ESR-Real)

Event Cascade Pruning (ECP) uses event data to guide temporal sampling, motion saliency filtering, and attention–event rank fusion. This repository provides the ESR-Real evaluation script, the customized Qwen2.5-VL implementation, and its visual processing tools.

## Installation

Use Python 3.10 and a CUDA GPU. Run the following commands from the repository root:

```bash
conda create -n eventprune python=3.10 -y
conda activate eventprune
python -m pip install -r requirements.txt
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
```

Prepare a local Qwen2.5-VL-7B-Instruct checkpoint. The customized `qwen_vl_utils` module is included in this repository.

## Data preparation

Download [ESR-Real](https://huggingface.co/datasets/pengtaoma00/ESR-Real) manually. It contains RGB videos, corresponding event data, and spatial QA annotations. Keep this layout:

```text
ESR-Real/
  test.jsonl
  videos/
    VIDEO_ID.mp4
  event/
    VIDEO_ID/
      *.pt
```

## Evaluation

Replace the model and data paths below with your local paths:

```bash
CUDA_VISIBLE_DEVICES=0 python esrbench.py \
  --ckpt_path /path/to/Qwen2.5-VL-7B-Instruct \
  --task_jsonl /path/to/ESR-Real/test.jsonl \
  --video_dir /path/to/ESR-Real/videos \
  --event_voxel_dir /path/to/ESR-Real/event \
  --result_dir outputs/esr_20 \
  --run_name esr_20_ECP \
  --num_fixed_frames 64 \
  --sampling_mode fixed_frames \
  --sampling_strategy ets \
  --ets_threshold 0.1 \
  --fps 1 \
  --emsf_drop_fraction 0.3 \
  --pruning_config_file configs/esr_20.json \
  --use_earf
```

## License

Code is licensed under Apache-2.0; upstream notices are retained in [NOTICE](NOTICE) and the source files. ESR-Real data is separately licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

## Citation

If you use our code or dataset in your research, please cite our paper:

```bibtex
@misc{ma2026eventprunecascadedeventassistedtoken,
  title={EventPrune: Cascaded Event-Assisted Token Pruning for Efficient First-Person Dynamic Spatial Reasoning},
  author={Pengtao Ma and Ziliang Zhou and Ciyu Ruan and Haoyang Wang and Kaiyuan Li and Zihang Gong and Wenhua Ding and Chen Gao and Jingao Xu and Xinlei Chen},
  year={2026},
  eprint={2605.19506},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2605.19506},
}
```
