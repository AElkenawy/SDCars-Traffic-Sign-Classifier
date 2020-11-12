# **Traffic Sign Classifier**

## Project goal
Design a Traffic Sign Classifier based on a Convolutional Neural Network architecture of LeNet-5 enhanced by some modifications to improve its performance in the classification task.

## Project steps
1. Dataset Summary and Exploration

    | Training set size  |  34799  |
    |--------|--------|
    |**Validation set size** | **4410** |
    |**Testing set size** | **12630** |
    | **Traffic sign image shape** | **32x32 pixels with 3 channels (RGB)** |
    | **Number of unique Classes/Labels** | **43** |

2. Designing, training and testing of the Model Architecture
    * Dataset is preprocessed using _grayscale_ and min-max _normalize_ routines.
    <img src="./imgs/3_sample.PNG" alt="Preprocessed sample">
    
    * Base-line CNN used is LeNet-5 convolutional network
    <img src="./imgs/4_lenet.PNG" alt="LeNet-5 architecture" width="550" height="300">
    
    **Utilized architecture**

    | Layer  |  Description  |
    |--------|--------|
    |Input | 32x32x1 Grayscale image|
    |Convolution 5x5 |1x1 stride, valid padding, outputs 28x28x6 |
    | ReLU (activation layer)| Rectifier linear unit |
    | Max pooling | 2x2 stride, outputs 14x14x6|
    |Convolution 5x5 | 1x1 stride, valid padding, outputs 10x10x16 |
    |ReLU (activation layer) | Rectifier linear unit |
    | Max pooling | 2x2 stride, outputs 5x5x6 |
    | Fully connected| Inputs 400, outputs 120|
    |ReLU (activation layer)| Rectifier linear unit |
    | Dropout ||
    |Fully connected | Outputs 84 |
    | ReLU (activation layer)| Rectifier linear unit |
    | Dropout ||
    | Fully connected |Outputs 43|
    | Dropout ||
    |Logits | 43 Logits/labels |
    
    * Introduced three Dropout layers before the fully connected layers resulted in a better _Validation accuracy_ of around 95%, instead of interrupted learning after the 5th epoch.
   
    **Model training hyperparameters**
    | Hyperparameter  |  Value  |
    |--------|--------|
    |Number of Epochs | 23|
    |Batch size |80 |
    | Learning rate α| 0.001 | 
    
3. Model testing on new images
    * Five web images of Traffic signs  are used to test system accuracy
    <img src="./imgs/7_web_images" alt="Web traffic signs images" width="550" height="300">
    
    |Image (Class ID)|Prediction (Class ID)|Classification Correctness|Training examples|
    |--------|--------|--------|--------|
    |Yield (13)|Yield (13)|Correct|1980|
    |Ahead only (35) |Ahead only (35)|Correct |1080|
    |Slippery road (23)|Right-of-way at the next intersection (11)|**Incorrect** |450|
    |Stop (14)|Stop (14)|Correct|690|
    |Right-of-way at the next intersection (11)| Dangerous curve to the right (20)|**Incorrect**|1170|

4. Softmax probabilities analysis of the new images
    * _tf.nn.softmax_ is used to review the 5 top probabilities related to each classified sign

### Basic Build Instructions
* German Traffic Sign dataset (used to train the model) can be found in [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
* Download [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)
* The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.
* the main program can be run by doing the following from the project top directory.
```sh
    git clone https://github.com/AElkenawy/SDCars-Traffic-Sign-Classifier
    cd CarND-Traffic-Sign-Classifier-Project
    jupyter notebook main.ipynb 
``` 

The jupyter notebook _main.ipynb_ is containing necessary routines for signs classification. The Traffic signs to be tested is located in  _./test_images and the classified labels are included inside the notebook (a _main.html_ file of the notebook is included inside project top directory)