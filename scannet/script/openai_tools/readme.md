Here we define tools used to generate openset graphs.

# One Quick Script

```bash
python scenes_inference_all.py --processed_data_folder xxx --raw_images_folder xxx --start_scene_id 0 --end_scene_id 100 --process_every_n_frames 1000 --output_folder xxx
```
Arguments are:
- __processed_data_folder__:  folder used to find the image ids to process. We search for png image names in this folder to get the ids. The images are generated in fixed-set map and are not used here (only names are used to maintain the same image ids). Standard name ```process/scans```.

-__raw_images_folder__: folder containing the raw RGB images from Scannet.

-__start_scene_id__: Start scene id in int. The scan id is omitted. When start_scene_id is 0, scene0000_00 and scene0000_01, etc. will be processed.

-__end_scene_id__: End scene id in int.

-__process_every_n_frames__: Number to control how many images will be saved to one batch. The number can't be too large because the max size of the batch jsonl for openai is 200M.

-__output_folder__: Output folder path to save the results. Standard name ```process/openset_scans```. The final result json files will be in ```process/openset_scans/sceneid/refined_instance```.

# Step by Step Script

## Prepare data
Change folder paths and start and end ID in ```write_batch_jsonl.py```, then run
```bash
python write_batch_jsonl.py 
```
A ```batch_inputs.jsonl``` file containing requests to openai and a ```request_image_paths.txt``` containing the images paths in the requests will be stored. Check the script to know the input arguments.

__Note__: Argument processed_data_folder is only used to find the image ids to process. We search for png image names in this folder to get the ids. The images are generated in fixed-set map and are not used here (only names are used to maintain the same image ids).

__NOTE__: the max size of the jsonl is 200M.

## Submit data
Run the following to upload the file to opebai's server and submit the batch.
```bash
python submit_batch_to_openai.py --input_file batch_inputs.jsonl
```
There will be a Input file ID and a Batch ID returned, e.g. ```file-Kmo9anv1bCece1tpQ6iTug```,```batch_6879163a500c819094599c98b52f8208```
We save Batch id in batch_id.txt.

## Check data status and retrieve
```bash
python check_batch_status_and_retrieve.py --batch_id xxx --check_interval 300
```
This will check if the result is ready every 300 seconds. If ready, the result will be retrieved and save at ```batch_results.jsonl```.


## Decode the result to get json annotation for each frame
Run
```bash
python decode_batch_results --output_folder xxx/processed/openset_scans
```
