# ASL_539_Project

Repository for ECE/CS 539 at UW-Madison's final project.

The repository contains code for ASL sign language detection and classification. 

Dataset: https://public.roboflow.com/object-detection/american-sign-language-letters

## How to run:

First make sure you have PyTorch installed. Any version would be fine. See https://pytorch.org/

E.g you can install with `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 `

The dataset is in `data` directory, with the train, test, validation split created in the subdirectories.

The analysis, including the whole workflow from data processing, training, evaluation, and the live demo, is in `asl_detection_1.ipynb` file. Just run through each cell for a whole workflow.

The analysis used in the project report running on colab can be found in the `colab` direcotry. Where `asl_detection_1.ipynb` is the training and evaluation for Faster R-CNN with ResNet50 pretrained backbone. Where as `copy_of_asl_detection_1.ipynb` is the training and evaluation for Faster R-CNN with Mobilenet V2 not pretrained backbone.
