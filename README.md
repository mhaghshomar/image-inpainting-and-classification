# image-inpainting-and-classification
This code trains a self-supervised deep learning model with a UNet to do an inpainting job on a set of unlabeled images (pool2). 
Then performes a classifing task on a set of labeled images (pool 1). lastly, it fine-tunes the weights of that on Pool 1 to train the final classifier. K-fold cross-validation is used to train the classifier. 
