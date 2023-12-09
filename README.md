# ASL_539_Project

Repository for ECE/CS 539 at UW-Madison's final project.

The repository contains code for ASL sign language detection and classification. 

Dataset: https://public.roboflow.com/object-detection/american-sign-language-letters

## How to run:

First make sure you have PyTorch installed. Any version would be fine. See https://pytorch.org/

E.g you can install with `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 `

The dataset is in `data` directory, with the train, test, validation split created in the subdirectories.

The analysis, including the whole workflow from data processing, training, evaluation, and the live demo, is in `asl_detection_1.ipynb` file. Just run through each cell for a whole workflow.
