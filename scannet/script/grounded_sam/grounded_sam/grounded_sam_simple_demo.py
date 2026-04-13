import inspect
import os
import sys
from pathlib import Path
from urllib.request import urlretrieve


def _bootstrap_repo_root() -> None:
    cur = Path(__file__).resolve()
    for parent in [cur.parent, *cur.parents]:
        if (parent / "scannet" / "script" / "thirdparty").is_dir():
            p = str(parent)
            if p not in sys.path:
                sys.path.insert(0, p)
            return


_bootstrap_repo_root()

# Lazily provision Grounded-Segment-Anything under scannet/script/thirdparty
from scannet.script.thirdparty.ensure_thirdparty import (
    add_to_syspath,
    ensure_grounded_sam,
)

path = str(ensure_grounded_sam(from_file=__file__))
print(path)
add_to_syspath(os.path.abspath(path))
# GroundingDINO lives under the cloned repo at GroundingDINO/
add_to_syspath(os.path.join(os.path.abspath(path), "GroundingDINO"))

from grounded_sam.groundingdino_ext_patch import apply_groundingdino_ms_deform_attn_patch

apply_groundingdino_ms_deform_attn_patch()

import cv2
import numpy as np
import supervision as sv
import torch
import torchvision
from groundingdino.util.inference import Model, preprocess_caption
from groundingdino.util.utils import get_phrases_from_posmap
from segment_anything import sam_model_registry, SamPredictor
from io import BytesIO


def _format_detection_labels(classes: list, detections) -> list:
    """Build one label string per detection (supervision no longer passes labels to BoxAnnotator)."""
    n = len(detections)
    labels = []
    for i in range(n):
        conf = float(np.asarray(detections.confidence[i]).reshape(()))
        raw_cid = detections.class_id[i]
        try:
            if raw_cid is None:
                name = "?"
            else:
                cid = int(np.asarray(raw_cid).reshape(()))
                name = classes[cid] if 0 <= cid < len(classes) else "?"
        except (TypeError, ValueError, IndexError):
            name = "?"
        labels.append(f"{name} {conf:0.2f}")
    return labels


def _annotate_boxes_and_labels(scene: np.ndarray, detections, labels: list) -> np.ndarray:
    box = sv.BoxAnnotator()
    label_cls = getattr(sv, "LabelAnnotator", None)
    if label_cls is not None:
        scene = box.annotate(scene=scene, detections=detections)
        return label_cls().annotate(scene=scene, detections=detections, labels=labels)
    if "labels" in inspect.signature(box.annotate).parameters:
        return box.annotate(scene=scene, detections=detections, labels=labels)
    return box.annotate(scene=scene, detections=detections)


def _ensure_checkpoint(url: str, dest_path: str) -> None:
    if os.path.exists(dest_path):
        return
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"[thirdparty] Downloading checkpoint to {dest_path}", file=sys.stderr)
    urlretrieve(url, dest_path)



class GroundedSam:
    def __init__(self) -> None:
        # Device
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # GroundingDINO config and checkpoint
        GROUNDING_DINO_CONFIG_PATH = path + "/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        GROUNDING_DINO_CHECKPOINT_PATH = path + "/groundingdino_swint_ogc.pth"
        # Segment-Anything checkpoint
        SAM_ENCODER_VERSION = "vit_h"
        SAM_CHECKPOINT_PATH = path + "/sam_vit_h_4b8939.pth"
        # Lazily download required checkpoints if missing
        _ensure_checkpoint(
            "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
            GROUNDING_DINO_CHECKPOINT_PATH,
        )
        _ensure_checkpoint(
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            SAM_CHECKPOINT_PATH,
        )
        # Building GroundingDINO inference model
        self.grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
        # Building SAM Model and SAM Predictor
        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(device=DEVICE)
        self.sam_predictor = SamPredictor(sam)

    @staticmethod
    def _predict_with_classes_and_token_logits(
        dino: Model,
        image_bgr: np.ndarray,
        classes: list,
        box_threshold: float,
        text_threshold: float,
    ):
        """
        GroundingDINO's `predict_with_classes` only returns Detections. This repo expects a
        per-detection feature vector for downstream JSON; we use the kept query token-logits
        (same filtering as upstream `predict`) of shape (n, 256).
        """
        caption = ". ".join(classes)
        caption = preprocess_caption(caption=caption)
        processed = Model.preprocess_image(image_bgr=image_bgr).to(dino.device)
        model = dino.model.to(dino.device)
        with torch.no_grad():
            outputs = model(processed[None], captions=[caption])
        prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]
        prediction_boxes = outputs["pred_boxes"].cpu()[0]
        keep = prediction_logits.max(dim=1)[0] > box_threshold
        logits_kept = prediction_logits[keep]
        boxes_kept = prediction_boxes[keep]
        tokenizer = model.tokenizer
        tokenized = tokenizer(caption)
        phrases = [
            get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace(".", "")
            for logit in logits_kept
        ]
        h, w, _ = image_bgr.shape
        detections = Model.post_process_result(
            source_h=h,
            source_w=w,
            boxes=boxes_kept,
            logits=logits_kept.max(dim=1)[0],
        )
        detections.class_id = Model.phrases2classes(phrases=phrases, classes=classes)
        return detections, logits_kept

    def generate_grid_points(self, box, num_points=9):
        """
        Generate a grid of points within the box, centered around the middle.
        num_points should be a perfect square (4, 9, 16, etc.)
        """
        x1, y1, x2, y2 = box
        grid_size = int(np.sqrt(num_points))
        
        # Calculate step sizes
        width = x2 - x1
        height = y2 - y1
        x_step = width / (grid_size + 1)
        y_step = height / (grid_size + 1)
        
        points = []
        # Generate grid points
        for i in range(1, grid_size + 1):
            for j in range(1, grid_size + 1):
                x = x1 + j * x_step
                y = y1 + i * y_step
                points.append([x, y])
        
        return np.array(points)

    # Prompting SAM with detected boxes
    def segment(self, sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        self.sam_predictor.set_image(image)
        result_masks = []

        for box in xyxy:
            # ADDED point prompt by CHG
            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2
            point_coords = np.array([[x_center, y_center]])
            point_labels = np.array([1])
            

            masks, scores, logits = sam_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)
    
    # Example: img_io contains the image as a BytesIO object
    # You can generate it using PIL
    def bytesio_to_cv2(self, img_io):
        """
        Convert an image in BytesIO to an OpenCV image.
        :param img_io: BytesIO object containing the image.
        :return: OpenCV image (numpy array).
        """
        # Read the bytes from BytesIO
        img_bytes = img_io.getvalue()
        
        # Convert the bytes to a numpy array
        np_array = np.frombuffer(img_bytes, np.uint8)
        
        # Decode the numpy array into an OpenCV image
        cv2_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)  # Use cv2.IMREAD_COLOR for color images
        return cv2_image
    
    def infer(self, image, classes, box_threshold, text_threshold, nms_threshold, confidence_threshold=0.3, image_type="cv2") -> np.ndarray:
        # load image
        if image_type == "cv2":
            pass
        elif image_type == "bytesio":        
            image = self.bytesio_to_cv2(image)
        else:
            raise ValueError(f"Invalid image type: {image_type}")
        # cv2.imshow("image", image)  # DO NOT USE THIS!!! Otherwise the following GPU task will be blocked
        # cv2.waitKey(0)
        # detect objects (+ token logits as features; upstream predict_with_classes is Detections-only)
        detections, features = self._predict_with_classes_and_token_logits(
            self.grounding_dino_model,
            image,
            classes,
            box_threshold,
            text_threshold,
        )
        # print(f"Detections: {detections}")
        # print(f"Features: {features}")

        # filter out detections with low confidence
        filter_ids = [i for i, confidence in enumerate(detections.confidence) if confidence > confidence_threshold]
        detections = detections[filter_ids]
        features = features[filter_ids]

        labels = _format_detection_labels(classes, detections)
        annotated_frame = _annotate_boxes_and_labels(image.copy(), detections, labels)
        # save the annotated grounding dino image
        # cv2.imwrite(path + "/" + "groundingdino_annotated_image.jpg", annotated_frame)

        # NMS post process
        #print(f"Before NMS: {len(detections.xyxy)} boxes")
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy), 
            torch.from_numpy(detections.confidence), 
            nms_threshold
        ).numpy().tolist()
        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]
        # print(f"After NMS: {len(detections.xyxy)} boxes")
        features = features[nms_idx] # only keep the features of the NMSed detections
        
        # convert detections to masks
        detections.mask = self.segment(
            sam_predictor=self.sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )
        mask_annotator = sv.MaskAnnotator()
        labels = _format_detection_labels(classes, detections)
        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = _annotate_boxes_and_labels(annotated_image, detections, labels)

        # save the annotated grounded-sam image
        # cv2.imwrite(path + "/" + "grounded_sam_annotated_image.jpg", annotated_image)

        if image_type == "bytesio":
            # Turn BGR to RGB
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        return annotated_image, detections.mask, detections.class_id, detections.confidence, features.cpu().numpy()
    
if __name__ == "__main__":
    # Example
    # image = cv2.imread(path + "/assets/beer_and_bread.jpg")
    # classes = ["A glass of beer", "Soft pretzels", "bowl of dip or spread", "greens"]
    image = cv2.imread("/media/cc/My Passport/dataset/scannet/images/scans/scene0261_00/frame-000003.color.jpg")
    classes = ["Bureau", "Cabinet", "Closet", "Counter Top", "Drawer", 
                "Equipment", "Hang", "Home Appliance", "Hook", 
                "Kitchen Utensil", "Pegboard", "Tool", "Toolbox", 
                "Utensil", "Workbench"]

    box_threshold = 0.25
    text_threshold = 0.25
    nms_threshold = 0.8

    grounded_sam = GroundedSam()
    annotated_image, masks, class_ids, confidences, features = grounded_sam.infer(image, classes, box_threshold, text_threshold, nms_threshold, confidence_threshold=0.3)


    print(masks.shape)
    print(class_ids)
    print(confidences)
    print(features.shape)

    for i in range(len(class_ids)):
        print(f"Class: {classes[class_ids[i]]}, Confidence: {confidences[i]}")

    cv2.imshow("annotated_image", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
