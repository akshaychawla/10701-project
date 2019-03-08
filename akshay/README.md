# Multilayer perceptron

## Reproduce results 

To reproduce the results in the paper for MLP, please run the following script: 

./marathon.sh  (WARNING THIS TAKES ~4 hours to run)

This will train 5 instances each of a 3 layer MLP (input, hidden and output) with 
the number of hidden layer neurons in {2k,4k,6k,8k,10k}. 

Then you can visualize all the results using tensorboard 

tensorboard --logdir="./runs/" 

## Other information 
The code python script is train_mlp.py. This script can be individually run as follows: 

python train_mlp.py 2000 ./runs/relu_2000/run1/ relu

This will train an MLP with 2000 hidden layer neurons with the ReLU activation functions. Training logs 
will be stored at ./runs/relu_2000/run1/ in the tensorboard format. 
