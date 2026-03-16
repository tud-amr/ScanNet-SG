# Instructions

## Generate Subscans
To randomly select a piece of data to generate a topolofical map and the point cloud for a subscan, run the following:
```bash
python generate_subscans.py --map_folder 'xxx/scannet/processed/scans/'  --output_map_folder 'xxx/scannet/subscans'
```
Check generate_subscans.py for input parameters.

## Generate Matching Data
After the Subscans are generated, run the following to generate pkl data for subscan-level scene graph registration.
```bash
python matcher_data_subscan.py --subscans_folder 'xxx/scannet/subscans' --processed_scans_folder 'xxx/scannet/processed/scans/' --data_output_dir desired_saving_folder
```

The result pkls and a summary csv will be saved.