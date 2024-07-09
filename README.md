> This is my implementation of the [traffic project](https://cs50.harvard.edu/ai/2023/projects/5/traffic/) from cs50AI.
> 
> Download the dataset [here](https://cdn.cs50.net/ai/2023/x/projects/5/gtsrb.zip).

### Model 1
I initially tried the model provided by the lesson for recognizing handwritten digits in the MNIST dataset,
but was not very successful.

### Model 2
I added another Conv2D layer and another MaxPooling2D layer, this time the accuracy was much higher.

### Model 3
Here I increased the contrast of all the images, so the NN would be able to recognize the edges of the street signs better.
I also changed all the activation functions to a Leaky ReLU (from the use of the ReLU).

### Model 4
I changed the optimizer from Adam to RMSProp and added a batch normalization layer after every other layer,
the model now learns faster and better.

### Model 5
In the penultimate layer I used a swish function instead of a leaky ReLU to further reduce overfitting,
this model seems to be more consistent and accurate than the previous one.

#### Training and evaluation
    Epoch 1/10
    500/500 [==============================] - 6s 10ms/step - loss: 0.8528 - accuracy: 0.7721
    Epoch 2/10
    500/500 [==============================] - 5s 11ms/step - loss: 0.1687 - accuracy: 0.9495
    Epoch 3/10
    500/500 [==============================] - 5s 11ms/step - loss: 0.1019 - accuracy: 0.9709
    Epoch 4/10
    500/500 [==============================] - 6s 11ms/step - loss: 0.0637 - accuracy: 0.9814
    Epoch 5/10
    500/500 [==============================] - 6s 11ms/step - loss: 0.0548 - accuracy: 0.9845
    Epoch 6/10
    500/500 [==============================] - 6s 12ms/step - loss: 0.0427 - accuracy: 0.9872
    Epoch 7/10
    500/500 [==============================] - 6s 12ms/step - loss: 0.0349 - accuracy: 0.9897
    Epoch 8/10
    500/500 [==============================] - 6s 12ms/step - loss: 0.0327 - accuracy: 0.9904
    Epoch 9/10
    500/500 [==============================] - 6s 12ms/step - loss: 0.0249 - accuracy: 0.9929
    Epoch 10/10
    500/500 [==============================] - 6s 13ms/step - loss: 0.0232 - accuracy: 0.9931
    333/333 - 1s - loss: 0.0348 - accuracy: 0.9904 - 1s/epoch - 4ms/step
