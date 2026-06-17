"""VisualOverload remotely-sourced FiftyOne Dataset Zoo dataset.

VisualOverload (CVPR 2026) is a visual question answering (VQA) benchmark of
2,720 question-answer pairs over 150 high-resolution public-domain paintings.
Ground-truth answers are held privately; models are scored via the official
evaluation server using each question's ``question_id``.

Source: https://huggingface.co/datasets/paulgavrikov/visualoverload

Each question becomes one FiftyOne sample (images are shared across the ~18
questions that reference them via duplicate ``filepath`` values).
"""

import ast
import glob
import json
import os

import fiftyone as fo

REPO_ID = "paulgavrikov/visualoverload"

# Question-level metadata columns read from the parquet shards. The parquet's
# embedded ``image`` bytes column is intentionally NOT read.
_QUESTION_COLUMNS = [
    "question_id",
    "question_type",
    "question",
    "options",
    "difficulty",
    "category",
    "default_prompt",
]


def _source_dir(dataset_dir):
    return os.path.join(dataset_dir, "source")


def download_and_prepare(dataset_dir, **kwargs):
    """Downloads the VisualOverload source files into ``dataset_dir/source``.

    Args:
        dataset_dir: the directory in which to download the dataset

    Returns:
        a ``(dataset_type, num_samples, classes)`` tuple. ``dataset_type`` is
        ``None`` to select FiftyOne's custom-loader path, which dispatches to
        :func:`load_dataset`.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            "The 'huggingface_hub' package is required to download "
            "VisualOverload. Install it via `pip install huggingface_hub`."
        ) from e

    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ImportError(
            "The 'pyarrow' package is required to read VisualOverload. "
            "Install it via `pip install pyarrow`."
        ) from e

    source_dir = _source_dir(dataset_dir)
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=source_dir,
    )

    # Count question rows from parquet metadata only (no column data is read)
    parquet_files = sorted(glob.glob(os.path.join(source_dir, "data", "*.parquet")))
    num_samples = sum(pq.ParquetFile(p).metadata.num_rows for p in parquet_files)

    return None, num_samples, None


def load_dataset(dataset, dataset_dir, **kwargs):
    """Builds one sample per VisualOverload question and adds them to ``dataset``.

    Args:
        dataset: the :class:`fiftyone.core.dataset.Dataset` to populate
        dataset_dir: the directory to which the dataset was downloaded
    """
    import pyarrow.parquet as pq
    from PIL import Image

    source_dir = _source_dir(dataset_dir)
    images_dir = os.path.join(source_dir, "images")

    # complexity.json provides the image-level info: which image each question
    # belongs to (the filepath link) and the image's win_rate. This mapping was
    # verified to match the parquet's per-row ``image.path`` exactly.
    with open(os.path.join(source_dir, "complexity.json")) as f:
        complexity = json.load(f)

    qid_to_image = {}
    image_to_win_rate = {}
    for entry in complexity:
        image = entry["image"]
        image_to_win_rate[image] = entry["win_rate"]
        for qid in entry["corresponding_qids"]:
            qid_to_image[qid] = image

    # Declare the schema explicitly so field types are deterministic regardless
    # of row ordering (e.g. an empty ``response_options`` list cannot be inferred).
    # question_type, difficulty, and category are NOT fields: they are stored as
    # prefixed sample tags (see the loop below).
    dataset.add_sample_field("question_id", fo.StringField)
    dataset.add_sample_field("question", fo.StringField)
    dataset.add_sample_field(
        "response_options",
        fo.ListField,
        subfield=fo.StringField,
        description=(
            "Answer options for 'choice' questions (e.g. ['yes', 'no']); empty "
            "otherwise. Listed as 'options' in the source dataset."
        ),
    )
    dataset.add_sample_field("default_prompt", fo.StringField)
    dataset.add_sample_field("image_id", fo.StringField)
    dataset.add_sample_field("win_rate", fo.FloatField)

    # Cache image dimensions so each of the 150 unique images is opened once.
    size_cache = {}

    def image_size(path):
        if path not in size_cache:
            with Image.open(path) as img:
                size_cache[path] = img.size  # (width, height)
        return size_cache[path]

    samples = []
    parquet_files = sorted(glob.glob(os.path.join(source_dir, "data", "*.parquet")))
    for parquet_file in parquet_files:
        table = pq.read_table(parquet_file, columns=_QUESTION_COLUMNS).to_pydict()
        for i in range(len(table["question_id"])):
            qid = table["question_id"][i]
            image = qid_to_image[qid]
            filepath = os.path.join(images_dir, image)
            width, height = image_size(filepath)

            options_raw = table["options"][i]
            response_options = (
                list(ast.literal_eval(options_raw)) if options_raw else []
            )

            # question_type / difficulty / category are stored as prefixed sample
            # tags. The prefixes disambiguate the "counting" and "ocr" values that
            # question_type and category both use.
            tags = [
                "test",
                f"question_type:{table['question_type'][i]}",
                f"difficulty:{table['difficulty'][i]}",
                f"category:{table['category'][i]}",
            ]

            samples.append(
                fo.Sample(
                    filepath=filepath,
                    tags=tags,
                    metadata=fo.ImageMetadata(width=width, height=height),
                    question_id=qid,
                    question=table["question"][i],
                    response_options=response_options,
                    default_prompt=table["default_prompt"][i],
                    image_id=os.path.splitext(image)[0],
                    win_rate=image_to_win_rate[image],
                )
            )

    dataset.add_samples(samples)

    dataset.description = (
        "VisualOverload (CVPR 2026): 2,720 VQA pairs over 150 high-resolution "
        "public-domain paintings. Ground-truth answers are held privately; "
        "evaluate via the official server using question_id."
    )
    dataset.info = {
        "source": "https://huggingface.co/datasets/paulgavrikov/visualoverload",
        "paper": "https://arxiv.org/abs/2509.25339",
        "project_page": "https://paulgavrikov.github.io/visualoverload/",
        "leaderboard": "https://huggingface.co/spaces/paulgavrikov/visualoverload-submit",
        "license": "cc-by-sa-4.0",
    }
    dataset.save()
