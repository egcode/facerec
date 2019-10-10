# Face Recognition using Pytorch

This is a Pytorch implementation of the face recognizer described in the papers:
* [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/pdf/1801.07698.pdf). (recomended)
* [CosFace: Large Margin Cosine Loss for Deep Face Recognition](https://arxiv.org/pdf/1801.09414.pdf). 
* [A Discriminative Feature Learning Approach for Deep Face Recognition](https://ydwen.github.io/papers/WenECCV16.pdf). 

### Face detection using MTCNN
Face detection on images and live camera performed using Tensorflow MTCNN implementation. [Multi-task CNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html). 
A OpenCV C++ implementation can be found [here](https://github.com/egcode/mtcnn-opencv)


## Pre-trained models
| Model name      | LFW accuracy | Training dataset | Architecture |
|-----------------|--------------|------------------|-------------|
| [IR_50_MODEL_arcface_ms1celeb_epoch90_lfw9962](https://drive.google.com/open?id=1itqqp1EWf6sfi0K4i6QYBR_j3NS7gw2i) | 0.9962        | M1-Celeb    | [IR_50](https://github.com/egcode/facerec/blob/master/models/irse.py) |

NOTE: If you use any of the models, please do not forget to give proper credit to those providing the training dataset as well.


## Compatibility
The code is tested using Pytorch 1.0 under OSX 10.14.5 and Ubuntu 16.04 with Python 3.6. 

