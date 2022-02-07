# Deep-Object
## What is this?
This project implements two concatenated deep neural networks which perform joint object detection and depth estimation.
We used [Mask R-CNN](https://arxiv.org/pdf/1703.06870.pdf) for object detection network, and [MiDaS](https://arxiv.org/abs/1907.01341) network for depth estimation. This project was done for EE-SUT deep learning course.
## Dependencies
You can install the dependencies by running: `pip install -r requirements.txt`

## Training

### Networks Inintial Weights
Both of the networks(object detector and depth estimator) were pretrained on large dataseta. We fine-tuned them using [NYU Depth Dataset V2](NYU Depth Dataset V2) dataset.

### Data Augmentatoin
We used following transforms as augmentation:
- Random crop
- Random horizontal flip
- Brightness change
- Random affine rotation (5 to 10 degree)

### Loss and Metric Plots
#### Object Detector
##### Training
![Training Loss](http://ee.sharif.edu/~amin/static/Deep/loss_OD.png)
##### Validation
![Validation Accuracy](http://ee.sharif.edu/~amin/static/Deep/loss_OD_validation.png)

#### Depth Estimator
##### Training
![Training Loss](http://ee.sharif.edu/~amin/static/Deep/loss_DD.png)
##### Validation
![Validation Accuracy](http://ee.sharif.edu/~amin/static/Deep/loss_DD_validation.png)

#### Joint Object Detection and Depth Estimation BELU
| ![BELU-1](http://ee.sharif.edu/~amin/static/Deep/BELU-1.png) | ![BELU-2](http://ee.sharif.edu/~amin/static/Deep/BELU-2.png) |
| ------------- |-------------|
| ![BELU-3](http://ee.sharif.edu/~amin/static/Deep/BELU-3.png) | ![BELU-4](http://ee.sharif.edu/~amin/static/Deep/BELU-4.png) |

## Testing

You can see some test images in the following plots:
| Etimated Depth  | Detected Objects with Estimated Depth|
| ------------- |-------------|
| ![Img Depth 1](http://ee.sharif.edu/~amin/static/Deep/Final_DD_test_1.png) | ![Img 2](http://ee.sharif.edu/~amin/static/Deep/Final_DO_test_1.png) |
| ![Img Depth 1](http://ee.sharif.edu/~amin/static/Deep/Final_DD_test_7.png) | ![Img Depth 2](http://ee.sharif.edu/~amin/static/Deep/Final_DO_test_7.png) |

## GUI
You can use the networks by GUI. First, install the depenedncies and the run the follosing command from the project main directory:
 `ptyhon3 GUI/main.py`. You can load more images to the app and move between the results using the slider placed at the bottom of the app page. Note that you shoud first train the models and save them in the ./Models directory. For saving or loading the models, change the load_model and save_model boolians in the train.ipynb Configs section.
