# SJTU-CS420-Course-Project
This is the course project for SJTU CS420 machine learning.  

## Requirements
* pytorch & torchvision
* numpy
* advertorch

## About codes
* fer.py: load the dataset.
* preprocess_fer2013.py: preprocess the fer2013 dataset.
* utils.py: some simple functions using in the training phase.
* train.py: train the CNN-based model on FER2013. (for the basic assignment)
* attack_sampled_images.py: sample 5 images from the dataset and carry some attacks and defenses.
* adv_train.py: use data augmenting to train a more robust model against adversarial attacks.

## Usage
### Preprocessing
* Put fer2013.csv in the data folder. 
* python preprocess_fer2013.py.
### Train and test a model (basic assignment)
* python train.py --model VGG19 --bs 64. 
* If you want to change the network or use other settings, be free to change the command.
* The output includes the model accuracy on public test set and private test set. 
### Attack sampled images (additional assignment)
* python attack_sampled_images.py
### Train a more robust model against the attacks (additional assignment)
* python adv_train.py

### To directly test a trained model
* Download our trained models from https://drive.google.com/open?id=11vqBwhcnyz1ZviLLiQrAS9YNibZ-RdJi and put in the saved_model folder. 
* Write a simple python code to load the model (or slightly modify train.py) and run the PublicTest or PrivateTest function.

## Acknowledgement
Thanks to BorealisAI and Wujie1010 for providing some useful tools and codes. We have referenced some of their work and codes.
