import os
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
from utils.common_utils import CommonUtils
from utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
import json
import copy
from pycocotools import mask as masktool
import pickle
from tqdm import tqdm
import argparse
import traceback

import numpy as np
import torch

def merge_highest_scores(results, original_width, original_height, min_box_area=0.01):
    """
    Merge detection results by selecting, for each label, the detection with the highest score,
    but only consider detections with sufficiently large bounding boxes.
    
    Args:
        results (list of dict): List of results (dictionaries) where each dict contains:
            - 'scores': Tensor of shape (N,)
            - 'labels': List or tensor of labels of length N
            - 'boxes': Tensor of shape (N, 4) in [x_min, y_min, x_max, y_max] format.
        min_box_area (int, optional): Minimum area (in normalized_size^2) required for a bounding box 
                                      to be considered. Default is 0.01.
    
    Returns:
        list of dict: A list containing one dictionary with merged 'scores', 'labels', and 'boxes'.
    """
    # Initialize dictionaries to store highest scores and corresponding boxes for each label
    label_scores = {}
    label_boxes = {}
    
    # Get the first result dict from the list
    result = results[0]
    scores = result['scores'].cpu().numpy()  # Convert tensor to numpy array
    labels = result['labels']
    boxes = result['boxes'].cpu().numpy()    # Convert tensor to numpy array
    
    # Iterate through all items
    for score, label, box in zip(scores, labels, boxes):
        # Calculate width and height assuming box format is [x_min, y_min, x_max, y_max]
        width = box[2] - box[0]
        height = box[3] - box[1]
        box_area = (width / original_width) * (height / original_height)
        #print("box_area", box_area)
        #import ipdb; ipdb.set_trace()

        # Skip detection if bounding box area is too small
        if box_area < min_box_area:
            continue
        
        # If label not seen before or new score is higher, update the dictionaries
        if label not in label_scores or score > label_scores[label]:
            label_scores[label] = score
            label_boxes[label] = box
    
    # Sort by scores in descending order
    sorted_items = sorted(label_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Create new lists in sorted order
    new_labels = []
    new_scores = []
    new_boxes = []
    
    for label, score in sorted_items:
        new_labels.append(label)
        new_scores.append(score)
        new_boxes.append(label_boxes[label])
    
    new_scores = np.array(new_scores)
    new_boxes = np.array(new_boxes)
    
    # Return the results in the same format as input
    return [{
        'scores': torch.from_numpy(new_scores).to('cuda:0'),
        'labels': new_labels,
        'boxes': torch.from_numpy(new_boxes).to('cuda:0')
    }]

def process_video(video_predictor, image_predictor, grounding_model, processor, video_path, entities, device,out_dir, stride = 1, stride_step = 20):
    output_masks_path = os.path.join(out_dir,'track_masks.pkl')
    output_jsons_path = os.path.join(out_dir,'track_jsons.json')

    # Dictionaries to store masks and JSON data
    all_masks = {}
    all_json_data = {}
    # Create per-entity output paths
    # Initialize video predictor state for each entity
    inference_state = video_predictor.init_state(
        video_path=video_path, 
        offload_video_to_cpu=True,
        offload_state_to_cpu=True
    )
    step = stride *  stride_step # Frame sampling step

    sam2_masks = MaskDictionaryModel()
    PROMPT_TYPE_FOR_VIDEO = "mask"  # box, mask or point
    objects_count = 0
    frame_object_count = {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    ori_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ori_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate new dimensions that preserve the original aspect ratio.
    # Target total pixels = 1280 * 720 = 921600
    #target_pixels = 1280 * 720
    scale_factor = 1 #np.sqrt(target_pixels / (ori_width * ori_height))
    width = int(ori_width * scale_factor)
    height = int(ori_height * scale_factor)
    #print(f"Resizing frames from ({ori_width}, {ori_height}) to ({width}, {height})")

    """human_entities = []
    human_keywords=['man', 'woman', 'child', 'human', 'boy', 'girl']
    for keyword in human_keywords:
        if keyword in (' '.join(entities)).lower():
            human_entities.append(keyword)"""

    filtered_entities = [
        entity for entity in entities 
        if not any(sub in entity.lower() for sub in ["hand", "foot", "feet", "head"])
    ]
    #filtered_entities += human_entities

    # Construct the entity description
    entity_description = '. '.join(filtered_entities) + '.'

    # Iterate over frames for the current entity
    for start_frame_idx in tqdm(range(stride // 2, total_frames, step), desc=f"Processing {video_path}"):
        # Set the video capture to the desired frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
        
        # Read the frame
        ret, frame = cap.read()
        if not ret:
            print(f"Error reading frame {start_frame_idx} in {video_path}")
            continue
        
        # Resize frame to the new dimensions
        frame_resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        image = Image.fromarray(frame_rgb)
        
        # Generate a base name for the frame
        image_base_name = f"frame_{start_frame_idx:05d}"
        
        mask_dict = MaskDictionaryModel(promote_type=PROMPT_TYPE_FOR_VIDEO, mask_name=f"mask_{image_base_name}")
        
        # Process the current entity in the frame
        #entity_description = entity.lower() + '.'  # Ensure lowercase and ends with a dot
        inputs = processor(images=image, text=entity_description, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = grounding_model(**inputs)

        #
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.25,
            text_threshold=0.25,
            target_sizes=[image.size[::-1]]
        )

        #import ipdb; ipdb.set_trace()
        merged_results = merge_highest_scores(results, original_width=width, original_height=height, min_box_area=0.01)

        # Set image for SAM predictor
        image_predictor.set_image(np.array(image.convert("RGB")))

        # Process the detection results
        input_boxes = merged_results[0]["boxes"]  # Tensor of boxes
        OBJECTS = merged_results[0]["labels"]     # Labels corresponding to entities

        if len(input_boxes) == 0:
            continue  # No detections for this entity

        # Get masks from SAM
        masks, scores, logits = image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        # Ensure mask dimensions are consistent
        if masks.ndim == 2:
            masks = masks[None]
            scores = scores[None]
            logits = logits[None]
        elif masks.ndim == 4:
            masks = masks.squeeze(1)

        # Add masks to mask dictionary
        if PROMPT_TYPE_FOR_VIDEO == "mask":
            mask_dict.add_new_frame_annotation(
                mask_list=torch.tensor(masks).to(device), 
                box_list=torch.tensor(input_boxes), 
                label_list=OBJECTS
            )
        else:
            raise NotImplementedError("SAM 2 video predictor only supports mask prompts")

        #import ipdb; ipdb.set_trace()
        # Update masks and track objects
        objects_count = mask_dict.update_masks(
            tracking_annotation_dict=sam2_masks, 
            iou_threshold=0.8, 
            objects_count=objects_count
        )
        frame_object_count[start_frame_idx] = objects_count
        video_predictor.reset_state(inference_state)
        if len(mask_dict.labels) == 0:
            print(f"No object detected in frame {start_frame_idx} of {video_path} for entity {entity_description}, skipping.")
            continue
        video_predictor.reset_state(inference_state)

        for object_id, object_info in mask_dict.labels.items():
            frame_idx, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                    inference_state,
                    start_frame_idx,
                    object_id,
                    object_info.mask,
            )
        
        video_segments = {}  # Track masks for subsequent frames
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(
            inference_state, 
            max_frame_num_to_track=step, 
            start_frame_idx=start_frame_idx,
            stride = stride
        ):
            frame_masks = MaskDictionaryModel()
            
            for i, out_obj_id in enumerate(out_obj_ids):
                out_mask = (out_mask_logits[i] > 0.0)  # Thresholding
                object_info = ObjectInfo(
                    instance_id=out_obj_id, 
                    mask=out_mask[0], 
                    class_name=mask_dict.get_target_class_name(out_obj_id), 
                    logit=mask_dict.get_target_logit(out_obj_id)
                )
                object_info.update_box()
                frame_masks.labels[out_obj_id] = object_info
                frame_masks.mask_height = out_mask.shape[-2]
                frame_masks.mask_width = out_mask.shape[-1]

            video_segments[out_frame_idx] = frame_masks
            sam2_masks = copy.deepcopy(frame_masks)

        # Save tracking masks and JSON data for the current entity
        for frame_idx, frame_masks_info in video_segments.items():
            mask = frame_masks_info.labels
            mask_img = torch.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width)
            for obj_id, obj_info in mask.items():
                mask_img[obj_info.mask == True] = obj_id

            mask_img = mask_img.numpy().astype(np.uint8)
            all_masks[frame_idx] = mask_img
            all_json_data[frame_idx] = frame_masks_info

        torch.cuda.empty_cache()

    # Release the video capture object after processing all frames for the current entity
    cap.release()

    # Reverse tracking if required
    start_object_id = 0
    object_info_dict = {}
    for frame_idx, current_object_count in frame_object_count.items():
        if frame_idx != 0:
            video_predictor.reset_state(inference_state)
            json_data = all_json_data[frame_idx]
            mask_array = all_masks[frame_idx]
            
            for object_id in range(start_object_id+1, current_object_count+1):
                object_info_dict[object_id] = json_data.labels[object_id]
                video_predictor.add_new_mask(inference_state, frame_idx, object_id, mask_array == object_id)

        if start_object_id == current_object_count:
            continue
        start_object_id = current_object_count
        
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(
            inference_state, 
            max_frame_num_to_track=step*6, 
            start_frame_idx=frame_idx,
            stride = stride,
            reverse=True
        ):
            if out_frame_idx not in all_json_data:
                all_json_data[out_frame_idx] = MaskDictionaryModel()
            if out_frame_idx not in all_masks:
                all_masks[out_frame_idx] = np.zeros((frame_masks.mask_height, frame_masks.mask_width), dtype=np.uint8)


            json_data = all_json_data[out_frame_idx]
            mask_array = all_masks[out_frame_idx]
            
            # Merge reverse tracking masks with original masks
            for i, out_obj_id in enumerate(out_obj_ids):
                out_mask = (out_mask_logits[i] > 0.0).cpu()
                if out_mask.sum() == 0:
                    continue
                object_info = object_info_dict[out_obj_id]
                object_info.mask = out_mask[0]
                object_info.update_box()
                json_data.labels[out_obj_id] = object_info
                mask_array = np.where(mask_array != out_obj_id, mask_array, 0)
                mask_array[object_info.mask] = out_obj_id
            
            all_masks[out_frame_idx] = mask_array
            all_json_data[out_frame_idx] = json_data

    for k in all_masks.keys():
        all_masks[k] = CommonUtils.encode_multi_instance_mask(all_masks[k])
    for k in all_json_data.keys():
        all_json_data[k] = all_json_data[k].to_dict()
        
    all_json_data['stride'] = stride
    all_json_data['step'] = step
    all_json_data['video_name'] = video_path
    all_json_data["resized_width"] = width
    all_json_data["resized_height"] = height
    torch.cuda.empty_cache()

    with open(output_masks_path, 'wb') as f:
        pickle.dump(all_masks, f)

    with open(output_jsons_path, 'w') as f:
        json.dump(all_json_data, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description='Process Videos from JSON')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the video')
    args = parser.parse_args()

    # Set your device
    device = torch.device("cuda")

    # Retrieve compute capability (major, minor)
    cc_major, cc_minor = torch.cuda.get_device_capability(device)
    # Choose dtype based on GPU: A100 usually has a major version >= 8
    if cc_major >= 8:
        dtype = torch.bfloat16  # Use bf16 for A100
        print("Using bfloat16 for autocast on this device.")
    else:
        dtype = torch.float16   # Use fp16 for V100 and others
        print("Using fp16 for autocast on this device.")

    # Initialize models and environment
    torch.autocast(device_type="cuda", dtype=dtype).__enter__()


    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Initialize SAM 2 models
    sam2_checkpoint = "sam2.1_hiera_large.pt"
    model_cfg = "sam2.1_hiera_l.yaml"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    image_predictor = SAM2ImagePredictor(sam2_image_model)

    # Initialize Grounding DINO model from HuggingFace
    model_id = "IDEA-Research/grounding-dino-base" 
    processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    args.json_file = "/path/to/your/json/file.json"
    # Load the JSON file containing video entries
    with open(args.json_file, 'r') as jf:
        video_dict = json.load(jf)

    video_path = args.video_path
    
    video_info = video_dict[video_path]
    video_name = os.path.basename(video_path)
    if video_info is not None:
        try:
            entities = video_info["entities"]
        except:
            entities = []
        if entities == []:
            return
    else:
        return
    if not entities:
        print(f"No entities found for {video_path}, skipping.")
        return

    process_video(video_predictor, image_predictor, grounding_model, processor, video_path, entities, device)


class Sam(object):
    """docstring for Sam"""
    def __init__(self):
        self.device = torch.device("cuda")

        # Retrieve compute capability (major, minor)
        cc_major, cc_minor = torch.cuda.get_device_capability(self.device)
        # Choose dtype based on GPU: A100 usually has a major version >= 8
        if cc_major >= 8:
            dtype = torch.bfloat16  # Use bf16 for A100
            print("Using bfloat16 for autocast on this device.")
        else:
            dtype = torch.float16   # Use fp16 for V100 and others
            print("Using fp16 for autocast on this device.")
    
        # Initialize models and environment
        torch.autocast(device_type="cuda", dtype=dtype).__enter__()
    
    
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
        # Initialize SAM 2 models
        sam2_checkpoint = "sam2.1_hiera_large.pt"
        model_cfg = "sam2.1_hiera_l.yaml"
        #device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
    
        self.video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
        sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=self.device)
        self.image_predictor = SAM2ImagePredictor(sam2_image_model)
    
        # Initialize Grounding DINO model from HuggingFace
        model_id = "IDEA-Research/grounding-dino-base"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
    
    def detect_img(self, image_path, keyword, out_path):
        try:
            image = Image.open(image_path)
            self.image_predictor.set_image(np.array(image.convert('RGB')))
            input = self.processor(images=image, text=keyword, return_tensors='pt').to(self.device)
            with torch.no_grad():
                outputs = self.grounding_model(**input)
            #print(outputs)
            result = self.processor.post_process_grounded_object_detection(
                outputs,
                input.input_ids,
                box_threshold=0.4,
                text_threshold=0.3,
                target_sizes=[image.size[::-1]]
            )
            input_boxes = result[0]["boxes"].cpu().numpy()
            if input_boxes.size == 0:
                return
            masks, scores, logits = self.image_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
            if masks.ndim == 4:
                masks = masks.squeeze(1)
            print(f"making dir{out_path}")
            os.makedirs(out_path, exist_ok=True)
            bboxes_path = os.path.join(out_path, 'track_jsons.json')
            masks_path = os.path.join(out_path, 'track_masks.npy')

            with open(bboxes_path, 'w') as fs:
                result[0]['scores'] = result[0]['scores'].cpu().numpy().tolist()
                result[0]['boxes'] = result[0]['boxes'].cpu().numpy().tolist()
                json.dump(result[0], fs, indent = 4)
            np.save(masks_path, np.array(masks))
        except Exception as e:
            print(f"错误: {str(e)}")
            print("完整堆栈信息:")
            traceback.print_exc()  # 打印完整的错误堆栈
            # 或者获取堆栈信息作为字符串
            stack_info = traceback.format_exc()
            # 可以将 stack_info 写入日志文件或进行其他处理
            pass

  
    def detect(self,video_path,json_path,out_dir):
        if not os.path.exists(json_path):
            print(f"JSON file {json_path} not found, detect human only!")
            entities = ["person"]
        else:
            with open(json_path, 'r') as jf:
                video_dict = json.load(jf)
            
            video_info = video_dict[list(video_dict.keys())[0]]

            if video_info is not None:
                try:
                    entities = video_info["entities"]
                except:
                    entities = []
                if entities == []:
                    return
            else:
                return
            if not entities:
                print(f"No entities found for {video_path}, skipping.")
                return
        
        process_video(self.video_predictor, self.image_predictor, self.grounding_model, self.processor, video_path, entities, self.device,out_dir)



if __name__ == "__main__":
    main()