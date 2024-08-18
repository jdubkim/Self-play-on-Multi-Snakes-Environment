# Competitive Self-play on Multi Snakes Game Environment

In this project, multi-snakes game environment and some famous Deep RL algorithms were implemented, and the agent was trained in this environment using self-play method. 
This project was proposed by OpenAI as one of the topics from ['Requests for Research 2.0'](https://blog.openai.com/requests-for-research-2/). 
The proposed research topic was as follows:  
1) Set up a reasonably large field with multiple snakes. 
2) Solve the environment using self-play with some RL algorithms, and observe what happens.  
   e.g. train current policy against a distribution of past policies. 
3) Inspect the learned behavior. 

Adding to this, inspired by 'slither.io', I added a new action that can cut other snake's body, so that when snakes are too big compared to the size of the environment, they can cut each other's body to stay much longer and get higher rewards.  

Multi-snakes OpenAI Gym environment was implemented based on [this code](https://github.com/nicomon24/Sneks).
This link was only used as a reference since the code in the link was not fully implemented. Deep RL algorithms implementations were written in tensorflow.

The slides are shared in this [link](https://docs.google.com/presentation/d/1lh0mDweE3k-gyRqgW-CUVC1bvujCZSVNyb7LDetOsW8/edit?usp=sharing).
  
This project is a result of [DeepLearning Camp Jeju 2018](http://jeju.dlcamp.org/2018/).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

You need tensorflow, gym and numpy to run this project. If you have your virtual env, please activate the env. 
```
pip3 install -r requirements.txt
```

### Installing

You need to install multi-snakes gym environment. 

```
cd gym-snake
pip3 install -e .
```

## Running the tests

You can run the test on several environments. Test codes are in 'test' folder. 
```
source activate tf # don't forget to activate virtual env
cd test
python3 multiple_test.py
```

## Built With

* [Tensorflow](https://www.tensorflow.org/) - ML Library used 
* [OpenAI Gym](https://maven.apache.org/) - Environment SetUp

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Sounam An - My mentor at DeepLearning Camp Jeju 2018
* Sourabh Bajaj - Mentor at DeepLearning Camp Jeju 2018
* Eric Jang - Mentor at DeepLearning Camp Jeju 2018

