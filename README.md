# Installing requirements
To make sure you can run the code, install all the dependencies from 'requirements.txt' with

```
python -m pip install -r requirements.txt
```

# Training the primary model
Training requires setting the path to the training csv and directory of training images.
Go to 'config.py' and set the variables 'CSV_TRAIN' and 'IMAGE_DIRECTORY_TRAIN' to the corresponding paths.

When the paths are set, run 'train_main_model.py' to train the model for 20 epochs and save the best model as
'best_model.pt'. To increase accuracy, go to 'train_main_model.py' and add the following arguments to the function call
 in the last line of the script: retrain_from='best_model.pt' or use the functional call below:

 ```
 train_model(model, transforms_train, transforms_cv, augmentations, retrain_from='best_model.pt', max_epochs=5,
 model_name='best_model2.pt')
 ```

This will save the new model as 'best_model2.pt'

# Inference
To make inferences, go to 'config.py' and set the variable 'CSV_TEST' to the csv file with the image ids of the images
you want to do inferences on. Set 'IMAGE_DIRECTORY_TEST' to the directory of the images to do inference on.

Run 'inference.py' to create a CSV with the inferences called 'inferences.csv'. This will use my best model
'RexNet9.pt'. To make inferences from your own model, go to 'inference.py' and change the variable 'model_name' to the
filename/path of some other model weights i.e. 'best_model2.pt'.

# Comparing additional non neural network models
Run 'non_neural_models.py' after setting the variables 'CSV_TRAIN' and 'IMAGE_DIRECTORY_TRAIN' in the 'config.py' file
to train and test a logistic regression classifier, gradient boosted trees, and support vector machine on the
training data.

