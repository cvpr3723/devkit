Plant Dataset Devkit
====================

This repository provides the devkit associtated with our benchmark. It contains tools for loading the data, computing the different metrics of every benchmark task, and scripts for generating the visualizations.

The folders contain the following contents:
- **evaluation_scripts**: scripts for computing the metrics of semantic segmentation, panoptic segmentation, leaf instance segmentation, hierarchical panoptic segmentation, and plant/leaf detection.
- **pdc_tools**: We provide a basic PyTorch-based dataloader for reading the data.
- **visualization_scripts**: Python scripts for visualization of the data and predictions that turns the provided ids in PNGs into masks.


