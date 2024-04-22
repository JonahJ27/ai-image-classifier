# Classifying Real Vs. Modified Vs. AI-Generated Images

The primary place to use this codebase is going to be in ```optim_visualizer.ipynb```

### Setup for an Initial Run

Cell 1: Imports. Run this to install all of the dependencies. If you are missing something when you try to run that cell, either navigate to the directory this is in and ```pip install [library]``` or you can do it directly in the notebook with ```!pip install [library]```

Cell 2: Our actual dataloader. Change the ```test_path``` and ```train_path``` to change which dataset you want to train on. You can also change the transforms applied fairly easily.

Cell 3: Trains the model. You can change the parameters by changing which model you import, the learning rate and the number of iterations. Loss function is Binary Cross Entropy Loss.

Cell 4: Uncomment this for the only convolutional layer model architecture. It is not very good and very slow to train.

Cell 5: The evaluation of the model. Pretty standard.

Cell 6: Loads data to try the model on images so we can get cool images

Cell 7: Tests the data loaded in cell 6. The first commented index predicts fake, but is actually real. The second predicts real, but is actually fake. The random generator can be used to just try random images in the test set.

Cell 8: Loads random images so you can view them later

Cell 9: Displays the images in a 2 x 2 grid so you can see images. Good for making some visualizations of the dataset.

### Models
Model 2 came about first, but is better so we gave it the higher number in the naming scheme.

```model_2.py```
The first model is architected with 3 sets of the following sequence of layers: a 3 x 3 convolutional layer, a ReLU, a BatchNorm2D layer and then a 2 x 2 max pooling layer. We then flatten, run through some linear layers with ReLU, BatchNorm1D and dropout inbetween before finishing with a sigmoid. This performs quite well.

```model.py```
Is the same after the flatten as the precious model, but only uses ReLU and convolutional layers before that. It alternates 3 x 3 and 5 x 5 layers with some of the 3 x 3s having padding of 1 pixel and a stride of 2. This model generally isn't very good.

```other models```
We've tried some other models that generally fall in between the accuracy of these two, but don't contribute to the end goal. The point of leaving these two is to show whata our final model looks like, and what a very bad model looks like to give context.

### Datasets

We have training sets of small, medium and full size for real and fake data. Small is 1000 images per class, medium is 5000 and the full one is 50000 images. We also have a dataset of images generated with Disco Diffusion to prove that you can use a different model to generate the images and then you can use it for a different dataset and still be accurate. We also have some images that are blurred or otherwise modified in specific spots to emulate image modification that you might see on social media. These are used to create datasets for classifying between real vs. edited images and AI-generated vs edited images.

You can change the dataloader path for test_path to any of the following:
- train
- train-medium
- train-small
- train-fake-edited
- train-real-edited

The first 3 correspond to ```test``` as the test path while, the last two correspond to:
- test-fake-edited
- test-real-edited

