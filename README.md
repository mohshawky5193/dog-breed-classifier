# Dog Breed Classifier
This project classifies dog breeds into one of **133** dog breeds using Pytorch

It includes a notebook with **6** parts <br/>
1. Human Face Detection(_Two methods used_)
2. Dog Detection(_Two pretrained models used_)
3. Classiying Dogs by training Neural Network From Scratch(Got **48%** accuracy)
4. Classifying Dogs By using pretrained model(Got **87%** accuracy)
5. Algorithm to detect human or dogs with the appropriate message
    * If dog detected then _predict the breed_
    * If human detected then _predict the most similar dog_
    * If neither print a message indication nothing detected
6. Testing the algorithm in step 5

# Installation
## Using conda
The file `environment.yaml` has all the necessary packages to install so all you need to do is

```
git clone https://github.com/mohshawky5193/dog-breed-classifier.git
cd dog-breed-classifier
conda env create -f environment.yaml
```
## Using pip
All the packages included in the file `requirements.txt` but you should modify the line for `torch == 1.0.0` and replace it with your version of `Pytorch` [here](https://pytorch.org/get-started/locally/) then copy and paste the link in the file `requirements.txt`

then do the following
```
git clone https://github.com/mohshawky5193/dog-breed-classifier.git
cd dog-breed-classifier
pip install -r requirements.txt
```
# Usage
After cloning and changing the directory run the app from the command line as follows
```
python dog-breed-classifier.py <input image file here> --output_path <output image file here>
```
_if no path specified the output image name will be `output.jpg`_

> * If the image is of human then it will lay a **snapchat like filter of dog** on the person's face like this<br/>
![Person Face Example](examples/human_out.jpg) 
> * If the image is of dog the it prints the dog breed in the console in addition to writing it on the dog image like this<br/>
![Dog Image Example](examples/dog_out.jpg)
> * If the image file contains neither human face nor a dog then print output message indicating so









# License
[MIT](https://choosealicense.com/licenses/mit/)