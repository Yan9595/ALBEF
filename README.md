## CS7643 Project: Visual Question Answering ##

### To run the experiments, download VQA 2.0 [image datasets](https://visualqa.org/download.html) and corresponding [annotations](https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/data.tar.gz), both put under `data` directory.
### The pretrained and finetuned checkpoints are provided by the author [here](https://github.com/salesforce/ALBEF#:~:text=Pre%2Dtrained%20checkpoint%20%5B14M%5D%20/%20%5B4M%5D).

### The notebooks mainly cover the experiments we focused on:
* ALBEF.ipynb: Evaluation of the given VQA checkpoint of ALBEF
* ALBEF_finetune.ipynb: Finetune and evaluate a downsized ALBEF by changing num of transformer blocks from 6 --> 3
* BLIP2.ipynb: Zero-shot evaluation of BLIP2 
* ALBEF_BLIP2.ipynb: Finetune and evaluate our idea of the combination of ALBEF and BLIP2.

