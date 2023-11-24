# FaceDetection in Python

The FacceDetection code is depend on https://github.com/biubug6/Pytorch_Retinaface

## Installation
### Clone and install
1. git clone git@github.com:ClearOneXu/FaceDetection.git

2. Codes are based on Python 3.10
3. Install requirements.
```Shell
pip install -r requirements.txt
```

### Run the code
The new detected face images will be saved in p_data/detected/, the txt file of the position of predicted images will be saved in p_data/pred/, and the ground truth should be put in p_data/ground_truth.

#### Program Input

The input to the program is a single argument either specifying a path to a folder of images or to a single image as follows. If there is no input image, the default images path will be p_data/photos/
```Shell
python script.py 1_faces_1.jpg
python script.py images/
```

#### Program Output
```Shell
1_faces_1.png 1
4_faces_2.png 4
3_faces_4.png 3
```


#### Evaluation metric
The evaluation metric contains Precision, Recall, F1-score and Average Precision. 
##### Program Input with Evaluation metric
```Shell
python script.py 1_faces_1.jpg -e
python script.py images/ -e
```
##### Program Output with Evaluation metric
```Shell
Precision: 0.9583
Recall: 1.0952
F1-score: 1.0222
Average Precision (AP) with IoU threshold 0.5: 0.7250
```