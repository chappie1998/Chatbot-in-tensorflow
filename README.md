# Chatbot-in-tensorflow

# Introduction:-
The chatbot is an A.I. powered software. A chatbot is often described as one of the most advanced and promising expressions of interaction between humans and machines. However, from a technological point of view, a chatbot only represents the natural evolution of a Question-Answering system leveraging Natural Language Processing (NLP). 

Sequence-to-sequence (seq2seq) models ([Sutskever et al.](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf), [2014, Cho et al., 2014](http://emnlp2014.org/papers/pdf/EMNLP2014179.pdf)) have enjoyed great success in a variety of tasks such as machine translation, speech recognition, and text summarization. This chatbot is based on seq2seq neural network with an attention mechanism. The GRU(Gated recurrent unit) cell is being used in the seq2seq NN. For the word embedding, Glove pretrained model is being used.

## Results:-
![alt text](https://github.com/ankitgc1/Chatbot-in-tensorflow/blob/master/images/Screenshot%20from%202019-10-18%2017-55-57.png)
## Dependencies:-
- Tensorflow >= 1.10
- matplotlib
- sklearn
- numpy
- unicodedata
- re
- os

## Download
- [Datasets](https://github.com/ankitgc1/Chatbot-in-tensorflow/tree/master/data)
- [pretrained model for embedding](https://www.kaggle.com/terenceliu4444/glove6b100dtxt)

### Deskcription:- 

The [Chatbot.py](https://github.com/ankitgc1/Chatbot-in-tensorflow/blob/master/chatbot.py) have Data preprocessing, Word embedding, Training, Evolution all the parts. I trained for three days on RTX2080ti on my own collected dataset and the results are pretty good. Change the parameter according to your machine configuration.  The seq2seq model used the GRU(Gated recurrent unit) layer. For the word embedding layer glove.6B.100d model is being used. After that encoder's output pass to the decoder. "Teacher forcing algorithm" - feeding the target as the next input. For the prediction "Greedy search algorithm" is being used.

##### Model Architecture:- 
![alt text](https://github.com/ankitgc1/Chatbot-in-tensorflow/blob/master/images/model_architecture.png)

##### GRU Architecture:-
![alt text](https://github.com/ankitgc1/Chatbot-in-tensorflow/blob/master/images/GRU.png)

