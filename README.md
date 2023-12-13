# ASL_539_Project

Repository for ECE/CS 539 at UW-Madison's final project on ASL sign language detection and classification.

The repository contains code for ASL sign language detection and classification. 

Dataset: https://public.roboflow.com/object-detection/american-sign-language-letters

## How to run:

First make sure you have PyTorch (preferably with cuda) installed. Any version should be fine. I developed my code based on 2.1.0 and 2.1.1 and both worked. See https://pytorch.org/

E.g you can install latest stable version with cuda 12.1 with `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 `

Other packages used are pretty standard there aren't any special ones (e.g pandas, matplotlib, tqdm etc).

The dataset is in `data` directory, with the train, test, validation split created in the subdirectories.

The analysis, including the whole workflow from data processing, training, evaluation, is in `asl_resnet50.ipynb` file for the Faster R-CNN with ResNet50 pretraine backbone, and `asl_mobilev2.ipynb` file for the Faster R-CNN with Mobilenet V2 backbone that is trained from scratch. Just run through each cell for a whole workflow.

To run the webcam application, run  `python3 app.py`. By default it takes the weight from Faster R-CNN ResNet50 pretrained if you download our weights and put in `models` directory.

## Trained Model

We provide the two trained model here. Cretae a `models` directory and put the trained model in there for inference.

Faster R-CNN with pretrained ResNet50 backbone: https://drive.google.com/file/d/12FsZ8kMy07ulYpl8IG8l2yicSz7nouCt/view?usp=sharing

Faster R-CNN with Mobilenet V2 backbone: https://drive.google.com/file/d/1iMNDi7Paik4X7_kbu94iXsfkIJ27NJ8M/view?usp=sharing

Weight used in presentation in class: https://drive.google.com/file/d/1RnwuNn87lEqj4c9QH3sIxYLgh2c_88VL/view?usp=sharing
