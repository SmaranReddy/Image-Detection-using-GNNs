import os
import json
import shutil

from PIL import Image

import spacy

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration
)

from pycocotools.coco import COCO


# ---------------------------------------------------
# PATHS
# ---------------------------------------------------

IMAGE_DIR = "data/coco/train2017"

ANNOTATION_FILE = (
    "data/coco/annotations/instances_train2017.json"
)

HALLUCINATION_DIR = "hallucinating/hal_images"
MULTI_DIR = "hallucinating/multiple_images"
SINGLE_DIR = "hallucinating/single_images"

HAL_JSON = "hallucinating/hal_groundtruth.json"
MULTI_JSON = "hallucinating/mult_groundtruth.json"
SINGLE_JSON = "hallucinating/sing_groundtruth.json"


# ---------------------------------------------------
# LIMITS
# ---------------------------------------------------

MAX_HALLUCINATION = 250
MAX_MULTI = 500
MAX_SINGLE = 250


# ---------------------------------------------------
# CREATE FOLDERS
# ---------------------------------------------------

os.makedirs(HALLUCINATION_DIR, exist_ok=True)
os.makedirs(MULTI_DIR, exist_ok=True)
os.makedirs(SINGLE_DIR, exist_ok=True)


# ---------------------------------------------------
# LOAD SPACY
# ---------------------------------------------------

print("Loading spaCy...")

nlp = spacy.load("en_core_web_sm")

print("spaCy loaded")


# ---------------------------------------------------
# LOAD BLIP
# ---------------------------------------------------

print("Loading BLIP...")

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

print("BLIP loaded")


# ---------------------------------------------------
# LOAD COCO
# ---------------------------------------------------

print("Loading COCO annotations...")

coco = COCO(ANNOTATION_FILE)

print("COCO loaded")


# ---------------------------------------------------
# CATEGORY MAP
# ---------------------------------------------------

cat_ids = coco.getCatIds()

cats = coco.loadCats(cat_ids)

category_map = {
    cat["id"]: cat["name"]
    for cat in cats
}


# ---------------------------------------------------
# EXTRACT NOUNS
# ---------------------------------------------------

def extract_nouns(text):

    doc = nlp(text)

    nouns = []

    for token in doc:

        if token.pos_ in ["NOUN", "PROPN"]:

            nouns.append(
                token.lemma_.lower()
            )

    return list(set(nouns))


# ---------------------------------------------------
# STORAGE
# ---------------------------------------------------

hallucination_results = []
multi_results = []
single_results = []


hallucination_count = 0
multi_count = 0
single_count = 0


# ---------------------------------------------------
# PROCESS IMAGES
# ---------------------------------------------------

image_ids = coco.getImgIds()

TOTAL_TARGET = (
    MAX_HALLUCINATION +
    MAX_MULTI +
    MAX_SINGLE
)

print(f"\nTarget images: {TOTAL_TARGET}")


for idx, image_id in enumerate(image_ids):

    # stop when all done
    if (
        hallucination_count >= MAX_HALLUCINATION
        and multi_count >= MAX_MULTI
        and single_count >= MAX_SINGLE
    ):
        break


    image_info = coco.loadImgs(image_id)[0]

    file_name = image_info["file_name"]

    image_path = os.path.join(
        IMAGE_DIR,
        file_name
    )

    print(f"\n[{idx}] {file_name}")


    # ------------------------------------------------
    # GET GT OBJECTS
    # ------------------------------------------------

    ann_ids = coco.getAnnIds(
        imgIds=image_id
    )

    anns = coco.loadAnns(ann_ids)

    gt_objects = []

    for ann in anns:

        cat_id = ann["category_id"]

        gt_objects.append(
            category_map[cat_id]
        )

    gt_objects = list(set(gt_objects))


    # ------------------------------------------------
    # SINGLE OBJECT IMAGES
    # ------------------------------------------------

    if (
        len(gt_objects) == 1
        and single_count < MAX_SINGLE
    ):

        shutil.copy(
            image_path,
            os.path.join(
                SINGLE_DIR,
                file_name
            )
        )

        single_results.append({

            "image": file_name,

            "objects": gt_objects
        })

        single_count += 1

        print(f"SINGLE [{single_count}]")

        continue


    # ------------------------------------------------
    # MULTI OBJECT IMAGES
    # ------------------------------------------------

    if (
        len(gt_objects) >= 2
        and multi_count < MAX_MULTI
    ):

        shutil.copy(
            image_path,
            os.path.join(
                MULTI_DIR,
                file_name
            )
        )

        multi_results.append({

            "image": file_name,

            "objects": gt_objects
        })

        multi_count += 1

        print(f"MULTI [{multi_count}]")


    # ------------------------------------------------
    # HALLUCINATION DETECTION
    # ------------------------------------------------

    if hallucination_count >= MAX_HALLUCINATION:
        continue


    try:

        image = Image.open(
            image_path
        ).convert("RGB")

    except:

        continue


    # generate BLIP caption
    inputs = processor(
        image,
        return_tensors="pt"
    )

    output = model.generate(**inputs)

    caption = processor.decode(
        output[0],
        skip_special_tokens=True
    )

    caption_objects = extract_nouns(
        caption
    )

    hallucinated = []

    for obj in caption_objects:

        if obj not in gt_objects:

            hallucinated.append(obj)


    # hallucination found
    if len(hallucinated) > 0:

        shutil.copy(
            image_path,
            os.path.join(
                HALLUCINATION_DIR,
                file_name
            )
        )

        hallucination_results.append({

            "image": file_name,

            "caption": caption,

            "ground_truth_objects": gt_objects,

            "hallucinated_objects": hallucinated
        })

        hallucination_count += 1

        print(f"HALLUCINATION [{hallucination_count}]")


# ---------------------------------------------------
# SAVE JSON FILES
# ---------------------------------------------------

with open(HAL_JSON, "w") as f:

    json.dump(
        hallucination_results,
        f,
        indent=4
    )


with open(MULTI_JSON, "w") as f:

    json.dump(
        multi_results,
        f,
        indent=4
    )


with open(SINGLE_JSON, "w") as f:

    json.dump(
        single_results,
        f,
        indent=4
    )


# ---------------------------------------------------
# FINAL REPORT
# ---------------------------------------------------

print("\nDONE\n")

print("Single images :", single_count)
print("Multi images :", multi_count)
print("Hallucinations :", hallucination_count)