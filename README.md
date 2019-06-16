# SJTU-CS420-Course-Project
This is the course project for SJTU CS420 machine learning. 

## Requirements
* pytorch & torchvision
* numpy
* advertorch

## Usage
### Preprocessing
* Preprocess the dataset. Put fer2013.csv in the data folder. 
* python preprocess_fer2013.py.
### Train and test a model (basic assignment)
* python train.py --model VGG19 --bs 64. 
* If you want to change the network or use other settings, be free to change the command.
* The output includes the model accuracy on public test set and private test set. 
### Attack sampled images (additional assignment)
* python attack_sampled_images.py
* The results will be stored.
### Train a more robust model against the attacks (additional assignment)
* python adv_train.py
* The trained model will be stored in the saved_models folder.
