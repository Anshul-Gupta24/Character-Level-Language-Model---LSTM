## Character Level Language Model using an LSTM

#### An implementation of a Character Level Language Model in Python using Numpy. For the purpose of understanding, everything including backpropagation has been coded form scratch.

#### The architecture includes an input layer, hidden layer and output layer.

#### The input is encoded using 1 hot encoding and is passed onto a hidden layer with 100 neurons. The output layer is a probability distribution over the vocabulary and an output character is sampled from this distribution. Backpropagation is carried out every 25 timesteps. </br> </br>

### Requirements
* Python 2.7
* Numpy
#### </br>
### Running the code

#### (Input is stored in input.txt)

#### ```>> python lstm.py```

#### Output is 200 character long samples of text every 100 iterations of backpropagation.

