# Tensorflow 2.0 vanilla DQN

A simple implementation of DQN with TF 2.0

## Getting Started

To run the project use the main.py file. 
To change between different OpenAI gym scenarios simply change the value in data/config/config.yml.
Moreover, other hyperparameters for each scenarios are modified via data/config/config_[scenario].yml configuration file. 

### Prerequisites
This project requires Python3 (tested on 3.6.8)



### Installing

GPU:
The requirements.txt file contain a tensorflow-gpu installation. 
All the basic nvidia drivers Cuda and cuDNN should be installed according to the official Tensorflow guide.   

CPU:
Change the 'tensorflow-gpu' to 'tensorflow' in the requirements.txt file.

Both: To install the requirements file:

```
pip3 install -r requirements.txt
```


## Refs
* [DQN](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=2ahUKEwir4Z6X4JblAhXMJVAKHel5CBYQFjAAegQIAxAC&url=https%3A%2F%2Fwww.cs.toronto.edu%2F~vmnih%2Fdocs%2Fdqn.pdf&usg=AOvVaw3M864Nv-fcyDpY-K9pPqEF) - DQN paper
* [Tensorflow 2.0](https://www.tensorflow.org/) - Deep learning framework
* [OpenAI - gym](https://gym.openai.com/) - Scenarios to run DQN on


## Contributing
As I used this project to learn how to transition from tf 1.X to 2.X the code could probably be improved.
Would be happy to receive PR with style \ syntax \ other scenarios.

As the goal of this project is to implement the simple DQN version, I am not interested in adding more complicated features at the moment.  

## Authors

* **Tom Jurgenson** - *Initial work* - [github](https://github.com/tomjur)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
