# udacity_model_re_training_deployment

Exercise: Model Re-Training and Deployment
Models are fun to build and explore, but they're only useful if we deploy them. We only want to deploy them if they're trained on the best, most recent data. In this exercise, we'll cover the basics of model re-training and deployment, including the following topics:

reading and interpreting records related to previously deployed ML models
re-training and saving ML models
pushing saved ML models to production
Starter File
We've provided a starter file called retraining.py, located in the /L2/ directory of your workspace below. You can add code to this file to create your solution.

Instructions - Reading Records of Previous Models
You'll start by reading a file called deployedmodelname.txt that contains the name of the previously deployed model. This is the model that you need to replace after doing retraining.

You should save the name of the model recorded in deployedmodelname.txt in a variable called deployedname in your Python script.

Next, you need to read a file called datalocation.txt. This file contains the filename of a new dataset that you'll read to accomplish model training.

You should save the filename recorded in datalocation.txt in a variable called datalocation in your Python script.