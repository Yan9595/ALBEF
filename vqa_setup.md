# VQA setup
- Download VQA v2 dataset and Visual Genome dataset from the original websites.
https://visualqa.org/download.html
Download balanced real images set, only coco dataset image is needed.

Note: coco dataset might have issues with security, use right click save link as.

In colab, data is placed in `./content/data`. 
Locally, you can place them in `./data`.

- Download and extract the provided dataset json files (Dataset json files for downstream tasks) and unzip it.

In configs/VQA.yaml, set the paths for the json files and the image paths.

- Download the pretrained models (from the readme file's link, we use `Pre-trained checkpoint [4M]`, and VQA checkpoint `Finetuned checkpoint for VQA`)
to './pretrained/vqa' folder (make new folder)

- Run single image evaluation
Use vqa_test_single_image.ipynb



