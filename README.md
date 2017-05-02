# Deep Reinforcement Learning for Playing Atari Games

Final project for BU EC500 K1/CS591 S2 Deep Learning

## Implemented Methods: ###
1. Deep Policy Network
2. Dueling Double Deep Q Network
3. Dueling Double Deep Q Network with LSTM
4. Asynchronous Advantage Actor Critic 
5. Asynchronous Advantage Actor Critic with GRU

## Requirement: ###
* Python 3.6 
* [Tensorflow 1.0](https://www.tensorflow.org/install/install_linux)
* [OpenAI Gym](https://github.com/openai/gym)

## To Run: ###
To train the model:
```bash
python3 main.py
```

To train other games or change model parameter, edit the corresponding params.py file.

## Reference: ###
* [Simple Reinforcement Learning with Tensorflow](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0)
* [Human-level control through deep reinforcement learning](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html)
* [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)
* [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)