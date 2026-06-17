# VisualOverload — FiftyOne Remote Zoo Dataset

<div align="center">
  <img width="1440" height="640" alt="VisualOverload sample paintings" src="https://github.com/user-attachments/assets/0258ed1b-8a8b-4008-a4ca-0c22caae5221" />
</div>

A [remotely-sourced FiftyOne Dataset Zoo](https://docs.voxel51.com/dataset_zoo/remote.html)
integration of [**VisualOverload**](https://huggingface.co/datasets/paulgavrikov/visualoverload)
(CVPR 2026).

VisualOverload is a visual question answering (VQA) benchmark comprising **2,720
question–answer pairs** over **150 high-resolution public-domain paintings** that
are densely populated with multiple figures, actions, and unfolding subplots. It
probes whether VLMs can perform simple, knowledge-free vision tasks in overloaded
scenes. Ground-truth answers are **held privately** — models are scored via the
[official evaluation server](https://huggingface.co/spaces/paulgavrikov/visualoverload-submit)
using each question's `question_id`.

## Installation

```bash
pip install -U fiftyone huggingface_hub pyarrow
```

## Usage

```python
import fiftyone.zoo as foz

# Loads (and downloads on first use) the dataset
dataset = foz.load_zoo_dataset("Voxel51/VisualOverload")

session = foz.fo.launch_app(dataset)
```

> If the integration is hosted at a different GitHub location, pass that URL
> instead, e.g. `foz.load_zoo_dataset("https://github.com/<user>/<repo>")`.

## What you get

One **sample per question** (2,720 samples; each painting is shared by the ~18
questions that reference it). Fields:

| Field | Type | Description |
|-------|------|-------------|
| `filepath` | image | Path to the painting (shared across its questions) |
| `question_id` | `StringField` | Unique id — the key used for leaderboard submissions |
| `question` | `StringField` | The question about the image |
| `response_options` | `ListField(StringField)` | Answer options for `choice` questions (e.g. `["yes", "no"]`); empty otherwise. Listed as `options` in the source dataset. |
| `default_prompt` | `StringField` | Ready-to-use prompt (question + options + format constraint) |
| `image_id` | `StringField` | Painting id (filename stem) — groups an image's questions |
| `win_rate` | `FloatField` | Per-image model win-rate from the benchmark (a difficulty signal) |
| `metadata` | `ImageMetadata` | Image width/height |

### Sample tags

`question_type`, `difficulty`, and `category` are encoded as **prefixed sample tags**
(filter on them in the App sidebar / `dataset.match_tags(...)`):

| Tag prefix | Values |
|------------|--------|
| `question_type:` | `choice`, `counting`, `ocr` |
| `difficulty:` | `easy`, `medium`, `hard` |
| `category:` | `activity`, `attributes`, `counting`, `ocr`, `reasoning`, `scene` |

Prefixes are used because `question_type` and `category` share the values `counting`
and `ocr` — bare tags would be ambiguous. Every sample is also tagged `test` (the
dataset's single, private-ground-truth split).

```python
hard_ocr = dataset.match_tags(["difficulty:hard", "question_type:ocr"], all=True)
```

### Running a VLM

Because each sample carries one `question`, a VQA model can read the prompt from
that field and write one prediction per sample:

```python
# Example shape — model that reads the question from a sample field
model.needs_fields = {"prompt": "question"}      # or use the model's prompt_field arg
dataset.apply_model(model, label_field="prediction")
```

To submit, export `question_id` + your `prediction` to the JSON format described
on the [dataset card](https://huggingface.co/datasets/paulgavrikov/visualoverload).

## License & citation

Dataset license: **CC BY-SA 4.0** (images are royalty-free public-domain artwork, CC0).

```bibtex
@InProceedings{Gavrikov_2026_visualoverload,
  author    = {Paul Gavrikov and Wei Lin and M. Jehanzeb Mirza and Soumya Jahagirdar and Muhammad Huzaifa and Sivan Doveh and Serena Yeung-Levy and James Glass and Hilde Kuehne},
  title     = {{VisualOverload}: Probing Visual Understanding of VLMs in Really Dense Scenes},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2026}
}
```
