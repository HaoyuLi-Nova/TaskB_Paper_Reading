#!/usr/bin/env python3
import os
import json
import argparse
import multiprocessing
from tqdm import tqdm
import csv
from functools import partial

def process_llm_output(text, original_caption, return_person_only=False):
    """
    Process the text output from the LLM.
    The input must contain both "ENTITIES:" and "INTERACTIONS:" sections.
    """
    try:
        text = text.replace(" - ","-")
        original_caption = original_caption.replace(" - ","-")
        # Check if both sections exist
        if 'ENTITIES:' not in text or 'INTERACTIONS:' not in text:
            raise ValueError("Input text must contain both ENTITIES and INTERACTIONS sections")
            
        sections = text.split('INTERACTIONS:')
        
        # Process entities section
        entities_section = sections[0].replace('ENTITIES:', '').strip()
        # Remove any bullet or dash and split per line
        entities = json.loads(entities_section) #[item.strip('* \n').replace('- ', '') for item in entities_section.split('\n') if item.strip('* \n')]
        
        if not entities:
            raise ValueError("No entities found in the input text")
        if return_person_only:
            new_entities = []
            for entity in entities:
                if not ('man' in entity.lower() or 'woman' in entity.lower() or 'girl' in entity.lower() or 'boy' in entity.lower() or 'elderly' in entity.lower() or 'child' in entity.lower()):
                    new_entities.append(entity)
            entities = new_entities
            entities.append("person")
        
        if len(entities) == 1:
            return None
        if len(entities) > 3:
            return None
        # Process interactions section
        interactions_section = sections[1].strip()
        interactions_raw = json.loads(interactions_section)
        
        #import ipdb; ipdb.set_trace()
        structured_interactions = []
        for interaction in interactions_raw:
            parts = interaction.split(' -> ')
            if len(parts) != 3:
                if len(parts) == 2:
                    structured_interactions.append({
                        "subject": parts[0].strip(),
                        "verb": parts[1].strip(),
                    })
                else:
                    print(f"Warning: Skipping invalid interaction format: {interaction}")
                    continue
            else:
                structured_interactions.append({
                    "subject": parts[0].strip(),
                    "verb": parts[1].strip(),
                    "object": parts[2].strip()
                })
            
        return {
            "entities": entities,
            "interactions": structured_interactions,
            "structured_caption": text,
            "caption": original_caption,
        }

    except Exception as e:
        print(f"Error processing text: {str(e)}")
        return None

def load_mapping_csv(mapping_file):
    """
    Load the mapping file (full_batch_2.json) where each line is a JSON dict.
    Build a dictionary mapping from video_id (the last part in the video path)
    to the part folder (e.g. "part_073").
    """
    mapping = {}
    caption_mapping={}
    with open(mapping_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader):
            """# Create a dictionary for each row with the keys 'caption' and 'video'
            sentences.append({
                'caption': row["text"],
                'video': row["file_name"]
            })"""
        
            
            video_path = row["video_path"]
            parts = video_path.split('/')
            part_folder = None
            for i in range(len(parts)):
                if parts[i].startswith("category"):
                    part_folder = [parts[i], parts[i+1]]
                    break
            video_id = parts[-1] if parts else None
            if video_id and part_folder:
                mapping[video_id] = part_folder
                caption_mapping[video_id] = row["caption"]
            else:
                print(f"Warning: Could not extract video id or part folder from {video_path}")
    return mapping, caption_mapping



def create_output_path(output_root, part_folder, custom_id):
    """
    Create the output path
    """
    base_id = custom_id.replace(".mp4", "")
    out_dir = os.path.join(output_root, part_folder[0], part_folder[1])#, subdir1, subdir2)
    os.makedirs(out_dir, exist_ok=True)
    filename = base_id + ".json"
    full_path = os.path.join(out_dir, filename)
    return full_path

CUSTOM_DETAILS = {}

def build_custom_details(mapping, caption_mapping):
    """
    Precompute a dictionary with keys as custom_id (with .mp4 appended if missing)
    and values as a tuple of (caption, part_folder). Only keys with a valid part_folder are kept.
    """
    details = {}
    # Combine keys from both mapping and caption_mapping.
    all_keys = set(list(mapping.keys()) + list(caption_mapping.keys()))
    for cid in all_keys:
        # Ensure the custom_id has a ".mp4" suffix.
        key = cid if cid.endswith(".mp4") else cid + ".mp4"
        caption_val = caption_mapping.get(key, "")
        part_folder_val = mapping.get(key)
        if part_folder_val is not None:
            details[key] = (caption_val, part_folder_val)
    return details

def init_pool(custom_details):
    """
    Pool initializer function.
    Set the global CUSTOM_DETAILS variable in each worker process.
    """
    global CUSTOM_DETAILS
    CUSTOM_DETAILS = custom_details

# ---------------------------
# Worker Function
# ---------------------------
def process_line_mp(line, output_root):
    """
    Process one JSONL line using a precomputed lookup from CUSTOM_DETAILS.
      - Parse the JSON.
      - Get the custom_id (append .mp4 if missing).
      - Retrieve the corresponding (caption, part_folder) from CUSTOM_DETAILS.
      - Process the raw caption content.
      - Create an output file path and write the final parsed output.
    Returns the output file path if successful, otherwise None.
    """
    if not line.strip():
        return None
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        print("Warning: Skipping invalid JSON line.")
        return None

    custom_id = data.get("custom_id")
    if ".mp4" not in custom_id:
        custom_id = custom_id + ".mp4"

    details = CUSTOM_DETAILS.get(custom_id)
    if details is None:
        print(f"Warning: custom_id {custom_id} not found in mapping file. Skipping...")
        return None

    caption, part_folder = details

    content = (
        data.get("response", {})
            .get("body", {})
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
    )

    output_file = create_output_path(output_root, part_folder, custom_id)
    # Skip if the output file already exists.
    if os.path.exists(output_file):
        return output_file

    parsed_output = process_llm_output(content, caption, return_person_only=True)
    if parsed_output is None:
        #print(f"Skipping custom_id {custom_id} due to processing error.")
        return None

    output_mp4_path = "your path"
    try:
        result = {output_mp4_path: parsed_output}
        with open(output_file, 'w', encoding='utf-8') as out_f:
            json.dump(result, out_f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error writing file {output_file}: {e}")
        return None
    return output_file

# ---------------------------
# Main Function
# ---------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Process LLM output and save parsed captions to JSON files using multiprocessing."
    )
    parser.add_argument("--llm_file", type=str, default="/path/to/your/jsonl",
                        help="Path to the LLM output JSONL file")
    parser.add_argument("--mapping_file", type=str, default="/path/to/your/csv",
                        help="Path to the mapping CSV file")
    parser.add_argument("--output_root", type=str, default="/path/to/your/output",
                        help="Root directory for saving output JSON files")
    parser.add_argument("--num_workers", type=int, default=multiprocessing.cpu_count(),
                        help="Number of workers to use for multiprocessing")
    parser.add_argument("--output_list_txt", type=str, default="/path/to/your/output.txt",
                        help="Text file listing processed items' MP4 paths")
    args = parser.parse_args()

    # Load mapping and caption data from CSV.
    mapping, caption_mapping = load_mapping_csv(args.mapping_file)
    print(f"Loaded {len(mapping)} video mapping entries.")

    # Precompute a dictionary for quick lookups in each worker.
    custom_details = build_custom_details(mapping, caption_mapping)

    # Read all the lines from the LLM output file.
    with open(args.llm_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    lines = lines

    # Use multiprocessing for processing the lines:
    with multiprocessing.Pool(args.num_workers, initializer=init_pool, initargs=(custom_details,)) as pool:
        # Create a partial to fix the output_root argument.
        process_func = partial(process_line_mp, output_root=args.output_root)
        results = list(tqdm(pool.imap(process_func, lines), total=len(lines), desc="Processing captions"))

    # Filter out failed or already processed items.
    processed_files = [r for r in results if r is not None]
    print(f"Finished processing. {len(processed_files)} files were saved or already existed.")

    # Prepare output list file: transform each processed file path to the corresponding MP4 path.
    txt_lines = []
    for output_file in processed_files:
        mp4_path = output_file
        txt_lines.append(mp4_path)

    # Write the list of processed MP4 paths to the text file.
    txt_file = args.output_list_txt
    with open(txt_file, 'w', encoding='utf-8') as ft:
        for line in txt_lines:
            ft.write(line + "\n")
    print(f"Written processed file paths to {txt_file}")

if __name__ == "__main__":
    main()