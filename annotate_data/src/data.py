import pathlib
from typing import Any

import pandas as pd
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers import PreTrainedModel  # type: ignore
from tqdm import tqdm

from annotate_data.src import config


def get_model_and_processors(model_name: str) -> tuple[PreTrainedModel, Any]:
    device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    return model.eval(), processor


@torch.inference_mode()
def detect_objects(
    model: PreTrainedModel,
    processor: Any,
    df: pd.DataFrame,
    batch_size: int,
    torch_dtype: torch.dtype = torch.float32,
) -> list[str]:
    """
    Detect objects in images using the provided model and processor.

    Args:
        model: The pre-trained model for object detection.
        processor: The processor to prepare images for the model.
        image_paths: List of paths to the images to be processed.

    Returns:
        List of detected objects as strings.
    """
    image_paths = df["image_path"].tolist()
    image_sizes = df["image_size"].tolist()
    all_detected_objects = []
    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i : i + batch_size]
        batch_image_sizes = image_sizes[i : i + batch_size]
        images = [
            Image.open(image_path).convert("RGB").resize(config.ObjectDectionModel.image_size)
            for image_path in batch_paths
        ]
        # Load and process images in batches
        inputs = processor(
            text=[config.ObjectDectionModel.prompt] * len(images),
            images=images,
            return_tensors="pt",
        ).to(model.device, torch_dtype)

        # Generate outputs for the current batch
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"].float(),
            max_new_tokens=128,
            do_sample=True,
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)

        parsed_answer = [
            processor.post_process_generation(line, task="<OD>", image_size=image_size)
            for line, image_size in zip(generated_text, batch_image_sizes)
        ]

        all_detected_objects.extend(parsed_answer)

    return all_detected_objects


def process_detected_objects(
    detected_objects: list[dict[str, dict[str, Any]]],
) -> tuple[list[list[str]], list[list[list[int]]]]:
    labels = [detected_object["<OD>"]["labels"] for detected_object in detected_objects]
    bboxes = [detected_object["<OD>"]["bboxes"] for detected_object in detected_objects]
    return labels, bboxes


def get_image_dataset():
    df = pd.read_csv(config.CSV_FILE)
    df["image_path"] = df["Sample_ID"].map(
        lambda x: pathlib.Path(f"./data/cats_and_dogs/data/{x}/image.jpg")
    )
    df["image_size"] = df["image_path"].map(lambda x: Image.open(x).size if x.exists() else (0, 0))

    model, processor = get_model_and_processors(config.ObjectDectionModel.model_name)
    detected_objects = detect_objects(
        model,
        processor,
        df,
        batch_size=config.ObjectDectionModel.batch_size,
        torch_dtype=torch.float16,
    )
    labels, bboxes = process_detected_objects(detected_objects)
    df["detected_labels"] = labels
    df["detected_bboxes"] = bboxes
    df.to_parquet("data/cats_and_dogs/detected_objects.parquet", index=False)


if __name__ == "__main__":
    get_image_dataset()
    # This will run the object detection and print the detected objects
    # You can modify this to save the results or process them further as needed
    print("Object detection completed.")
