# MangaBubbleSegmentation
Spring 2020 ML project

Use MangaScrub on google colab or jupyter notebook to download manga pages from https://rawdevart.com/

Use segmenter.py to split the images from Training and Grayscale to the sliding window images.

Manga Text Detection only runs on python2 and is used to produce the OCR mask (run _LocateTextAll.py after putting chapter folders in the _input. result is in _output)

Trained models and checkpoints are stored in Models folder.

Files ending in Train are for training
Files ending in Run are for generating heatmaps or other results
Files ending in Analyze are for checking accuracy and loss
