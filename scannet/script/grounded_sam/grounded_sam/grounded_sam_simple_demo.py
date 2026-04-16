import os
import sys

_here = os.path.dirname(__file__)
_candidates = [
    os.path.abspath(os.path.join(_here, "../../thirdparty/Grounded-Segment-Anything")),
    os.path.abspath(os.path.join(_here, "../../../thirdparty/Grounded-Segment-Anything")),
]
path = next((p for p in _candidates if os.path.isdir(p)), _candidates[0])
print(path)

# Ensure we import the vendored repos (avoid pip name conflicts).
sys.path.insert(0, os.path.join(path, "segment_anything"))
sys.path.insert(0, os.path.join(path, "GroundingDINO"))
sys.path.insert(0, path)
import cv2
import numpy as np
import supervision as sv
import torch
import torchvision
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
from io import BytesIO



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
        # Building GroundingDINO inference model
        self.grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
        # Building SAM Model and SAM Predictor
        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(device=DEVICE)
        self.sam_predictor = SamPredictor(sam)

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
    
    def infer(self, image, classes, box_threshold, text_threshold, nms_threshold, confidence_threshold=0.5, image_type="cv2") -> np.ndarray:
        # load image
        if image_type == "cv2":
            pass
        elif image_type == "bytesio":        
            image = self.bytesio_to_cv2(image)
        else:
            raise ValueError(f"Invalid image type: {image_type}")
        # cv2.imshow("image", image)  # DO NOT USE THIS!!! Otherwise the following GPU task will be blocked
        # cv2.waitKey(0)
        # detect objects
        detections, features = self.grounding_dino_model.predict_with_classes(
            image=image,
            classes=classes,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        # print(f"Detections: {detections}")
        # print(f"Features: {features}")

        # filter out detections with low confidence
        filter_ids = [i for i, confidence in enumerate(detections.confidence) if confidence > confidence_threshold]
        detections = detections[filter_ids]
        features = features[filter_ids]

        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        labels = [
            f"{classes[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _, _
            in detections
        ]
        annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)
        try:
            label_annotator = sv.LabelAnnotator()
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        except Exception:
            # Older/newer supervision variants may not have LabelAnnotator or may differ in signature.
            pass
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
        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        labels = [
            f"{classes[class_id]} {confidence:0.2f}" 
            for _, _, confidence, class_id, _, _ 
            in detections]
        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
        try:
            label_annotator = sv.LabelAnnotator()
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        except Exception:
            pass

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
