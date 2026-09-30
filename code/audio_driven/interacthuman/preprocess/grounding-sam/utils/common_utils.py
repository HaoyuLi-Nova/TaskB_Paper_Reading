import os
import json
import cv2
import numpy as np
from dataclasses import dataclass
import supervision as sv
import random
from pycocotools import mask as masktool

class CommonUtils:

    @staticmethod
    def compute_frame_mapping(data_dict, total_frames):
        """
        Precomputes a mapping for each frame index [0, total_frames) to the nearest available key in data_dict.
        It assumes that keys in data_dict can be converted to integers.
        
        Returns a dictionary mapping frame_idx -> available key from data_dict.
        """
        # Convert available keys to numeric form with a mapping to the original key type
        numeric_keys = {}
        for key in data_dict.keys():
            try:
                numeric_key = int(key)
                numeric_keys[numeric_key] = key  # Map the numeric key to the original key
            except ValueError:
                continue

        if not numeric_keys:
            return {}

        available_numeric_keys = sorted(numeric_keys.keys())
        
        mapping = {}
        for frame_idx in range(total_frames):
            # If the frame_idx directly has data in the dictionary, use it
            if frame_idx in numeric_keys:
                mapping[frame_idx] = numeric_keys[frame_idx]
            else:
                # Find the available key with the smallest absolute difference
                nearest = min(available_numeric_keys, key=lambda x: abs(x - frame_idx))
                mapping[frame_idx] = numeric_keys[nearest]
        return mapping
    
    @staticmethod
    def draw_masks_and_box_with_supervision_to_mp4(video_path, all_json_data, all_masks, output_video_path, frame_rate=None):
        # Open the input video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # If frame_rate is not provided, use the original video's frame rate
        if frame_rate is None:
            frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Since we will concatenate two frames side by side,
        # set the combined frame width to twice the original width.
        combined_width = width * 2

        # Set up the video writer for the combined frame (original + annotated)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (combined_width, height))
        
        # Precompute the mapping for json and mask data
        mapping_json = CommonUtils.compute_frame_mapping(all_json_data, total_frames)
        mapping_masks = CommonUtils.compute_frame_mapping(all_masks, total_frames)

        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                print(f"Error reading frame {frame_idx}")
                break

            # Use the precomputed mappings to get the nearest available data
            json_key = mapping_json.get(frame_idx)
            mask_key = mapping_masks.get(frame_idx)
            json_data = all_json_data.get(json_key) if json_key is not None else None
            mask_data = all_masks.get(mask_key) if mask_key is not None else None

            # Default annotation is just a copy of the original frame
            annotated_frame = frame.copy()

            if json_data and mask_data:
                mask_array = CommonUtils.decode_multi_instance_mask(mask_data)
                
                # Get unique ids from the mask array (ignoring the background id of 0)
                unique_ids = np.unique(mask_array)
                unique_ids = unique_ids[unique_ids != 0]
                
                # Extract individual masks for each non-zero id
                all_object_masks = []
                for uid in unique_ids:
                    object_mask = (mask_array == uid)
                    all_object_masks.append(object_mask[None])  # Adds an extra dimension (1, H, W)
                
                if len(all_object_masks) == 0:
                    print(f"No object detected in frame {frame_idx}, skipping annotation.")
                else:
                    # Concatenate masks to shape (n, H, W)
                    all_object_masks = np.concatenate(all_object_masks, axis=0)

                    all_object_boxes = []
                    all_object_ids = []
                    all_class_names = []
                    
                    # Process detected objects as provided in JSON data
                    for obj_id, obj_item in json_data['labels'].items():
                        class_name = obj_item['class_name']
                        instance_id = obj_item['instance_id']
                        # Ensure that the instance ID is present in the mask unique IDs
                        if instance_id not in unique_ids:
                            continue
                        
                        x1, y1, x2, y2 = obj_item['x1'], obj_item['y1'], obj_item['x2'], obj_item['y2']
                        all_object_boxes.append([x1, y1, x2, y2])
                        
                        all_object_ids.append(instance_id)
                        all_class_names.append(class_name)

                    # Order detections by their ids for consistency
                    paired = list(zip(all_object_ids, all_object_boxes, all_class_names))
                    sorted_pairs = sorted(paired, key=lambda pair: pair[0])
                    all_object_ids = [pair[0] for pair in sorted_pairs]
                    all_object_boxes = [pair[1] for pair in sorted_pairs]
                    all_class_names = [pair[2] for pair in sorted_pairs]

                    if len(all_object_ids) == 0 or len(all_object_boxes) == 0:
                        print(f"No matching object data in frame {frame_idx}, skipping annotation.")
                    else:
                        # Create detections for annotation using Supervision package
                        detections = sv.Detections(
                            xyxy=np.array(all_object_boxes),
                            mask=all_object_masks,
                            class_id=np.array(all_object_ids, dtype=np.int32),
                        )

                        # Create custom labels combining instance id and class name
                        labels = [
                            f"{instance_id}: {class_name}" 
                            for instance_id, class_name in zip(all_object_ids, all_class_names)
                        ]
                        
                        box_annotator = sv.BoxAnnotator()
                        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
                        
                        label_annotator = sv.LabelAnnotator()
                        annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
                        
                        mask_annotator = sv.MaskAnnotator(opacity=0.5)
                        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)

            # Combine the original and annotated views side-by-side
            combined_frame = np.concatenate((frame, annotated_frame), axis=1)
            out.write(combined_frame)

        # Release video capture and writer resources
        cap.release()
        out.release()

    @staticmethod
    def creat_dirs(path):
        """
        Ensure the given path exists. If it does not exist, create it using os.makedirs.

        :param path: The directory path to check or create.
        """
        try: 
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                print(f"Path '{path}' did not exist and has been created.")
            else:
                pass
                #print(f"Path '{path}' already exists.")
        except Exception as e:
            print(f"An error occurred while creating the path: {e}")

    @staticmethod
    def draw_masks_and_box_with_supervision(raw_image_path, mask_path, json_path, output_path):
        CommonUtils.creat_dirs(output_path)
        raw_image_name_list = os.listdir(raw_image_path)
        raw_image_name_list.sort()
        for raw_image_name in raw_image_name_list:
            image_path = os.path.join(raw_image_path, raw_image_name)
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError("Image file not found.")
            # load mask
            mask_npy_path = os.path.join(mask_path, "mask_"+raw_image_name.split(".")[0]+".npy")
            mask = np.load(mask_npy_path)
            # color map
            unique_ids = np.unique(mask)
            
            # get each mask from unique mask file
            all_object_masks = []
            for uid in unique_ids:
                if uid == 0: # skip background id
                    continue
                else:
                    object_mask = (mask == uid)
                    all_object_masks.append(object_mask[None])
            
            if len(all_object_masks) == 0:
                output_image_path = os.path.join(output_path, raw_image_name)
                cv2.imwrite(output_image_path, image)
                continue
            # get n masks: (n, h, w)
            all_object_masks = np.concatenate(all_object_masks, axis=0)
            
            # load box information
            file_path = os.path.join(json_path, "mask_"+raw_image_name.split(".")[0]+".json")
            
            all_object_boxes = []
            all_object_ids = []
            all_class_names = []
            object_id_to_name = {}
            with open(file_path, "r") as file:
                json_data = json.load(file)
                for obj_id, obj_item in json_data["labels"].items():
                    # box id
                    instance_id = obj_item["instance_id"]
                    if instance_id not in unique_ids: # not a valid box
                        continue
                    # box coordinates
                    x1, y1, x2, y2 = obj_item["x1"], obj_item["y1"], obj_item["x2"], obj_item["y2"]
                    all_object_boxes.append([x1, y1, x2, y2])
                    # box name
                    class_name = obj_item["class_name"]
                    
                    # build id list and id2name mapping
                    all_object_ids.append(instance_id)
                    all_class_names.append(class_name)
                    object_id_to_name[instance_id] = class_name
            
            # Adjust object id and boxes to ascending order
            paired_id_and_box = zip(all_object_ids, all_object_boxes)
            sorted_pair = sorted(paired_id_and_box, key=lambda pair: pair[0])
            
            # Because we get the mask data as ascending order, so we also need to ascend box and ids
            all_object_ids = [pair[0] for pair in sorted_pair]
            all_object_boxes = [pair[1] for pair in sorted_pair]
            
            detections = sv.Detections(
                xyxy=np.array(all_object_boxes),
                mask=all_object_masks,
                class_id=np.array(all_object_ids, dtype=np.int32),
            )
            
            # custom label to show both id and class name
            labels = [
                f"{instance_id}: {class_name}" for instance_id, class_name in zip(all_object_ids, all_class_names)
            ]
            
            box_annotator = sv.BoxAnnotator()
            annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)
            label_annotator = sv.LabelAnnotator()
            annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
            mask_annotator = sv.MaskAnnotator()
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
            
            output_image_path = os.path.join(output_path, raw_image_name)
            cv2.imwrite(output_image_path, annotated_frame)
            print(f"Annotated image saved as {output_image_path}")

    @staticmethod
    def encode_multi_instance_mask(mask):
        unique_labels = np.unique(mask)
        unique_labels = unique_labels[unique_labels != 0]  # Exclude background
        encodings = []
        for label in unique_labels:
            binary_mask = (mask == label).astype(np.uint8)
            rle = masktool.encode(np.asfortranarray(binary_mask))
            encodings.append((label, rle))
        
        # Save shape parameters
        shape_params = {'height': mask.shape[0], 'width': mask.shape[1]}
        if mask.ndim > 2:
            shape_params['depth'] = mask.shape[2]
        
        return {'mask': encodings, 'shape': shape_params}

    @staticmethod
    def decode_multi_instance_mask(encodings_dict):
        encodings = encodings_dict['mask']
        shape_params = encodings_dict['shape']
        # Reconstruct shape from parameters
        if 'depth' in shape_params:
            shape = (shape_params['height'], shape_params['width'], shape_params['depth'])
        else:
            shape = (shape_params['height'], shape_params['width'])
        
        mask = np.zeros(shape, dtype=np.uint8)
        for label, encoding in encodings:
            binary_mask = masktool.decode(encoding).astype(np.uint8)
            mask[binary_mask == 1] = label
        return mask
    @staticmethod
    def draw_masks_and_box(raw_image_path, mask_path, json_path, output_path):
        CommonUtils.creat_dirs(output_path)
        raw_image_name_list = os.listdir(raw_image_path)
        raw_image_name_list.sort()
        for raw_image_name in raw_image_name_list:
            image_path = os.path.join(raw_image_path, raw_image_name)
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError("Image file not found.")
            # load mask
            mask_npy_path = os.path.join(mask_path, "mask_"+raw_image_name.split(".")[0]+".npy")
            mask = np.load(mask_npy_path)
            # color map
            unique_ids = np.unique(mask)
            colors = {uid: CommonUtils.random_color() for uid in unique_ids}
            colors[0] = (0, 0, 0)  # background color

            # apply mask to image in RBG channels
            colored_mask = np.zeros_like(image)
            for uid in unique_ids:
                colored_mask[mask == uid] = colors[uid]
            alpha = 0.5  # 调整 alpha 值以改变透明度
            output_image = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)


            file_path = os.path.join(json_path, "mask_"+raw_image_name.split(".")[0]+".json")
            with open(file_path, 'r') as file:
                json_data = json.load(file)
                # Draw bounding boxes and labels
                for obj_id, obj_item in json_data["labels"].items():
                    # Extract data from JSON
                    x1, y1, x2, y2 = obj_item["x1"], obj_item["y1"], obj_item["x2"], obj_item["y2"]
                    instance_id = obj_item["instance_id"]
                    class_name = obj_item["class_name"]

                    # Draw rectangle
                    cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Put text
                    label = f"{instance_id}: {class_name}"
                    cv2.putText(output_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Save the modified image
                output_image_path = os.path.join(output_path, raw_image_name)
                cv2.imwrite(output_image_path, output_image)

                print(f"Annotated image saved as {output_image_path}")

    @staticmethod
    def random_color():
        """random color generator"""
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
