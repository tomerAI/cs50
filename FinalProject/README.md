# ML-model of Spaceship Titanic
#### Video Demo:  <https://youtu.be/kR9Akbpfk7E>
#### Description:
In this project i created a webpage that utilizes a ML-model that predicts if a passenger is transported to an alternate dimension or not. The user can create a passenger and see if they were transported. 

Spaceship.py uses data from ./data and trains the ML-model and saves the model into a gb-model.pkl file. This file must be run before the app.py or else the webpage will not function.

App.py uses the gb-model.pkl file to predict a passenger. App.py also receives input from the user and transforms the data so that it fits the model saved to gb-model.pkl. 

./templates hold the html files
Layout.html is the layout for the webpage. It uses bootstrap and has a javascript function that alerts the user of wrong usage. 
Predict.html receives inputs from the user and sends them to App.py

feature_columns and feature_names are files created in Spaceship.py that helps transform the data input from the user and also provides the foundation of the data output to the user. 

Hisory.db saves all of the history of all passengers created by the user. 

requirements.txt holds the libraries used to create the project



The brain to predict if a passenger survives is created in the spaceship.py file. It creates a .pkl file that the app.py must have access to.
The ML-model takes data from the kaggle's competition called spaceship titanic which is then processed into a new dataset. Some of the data is removed due to high cardinality and if there is no believed causality. One data column is also split into 2 data columns to better understand the data's correlation. The data consist of numerical values and categories and the categories must be transformed into numerical values, this is achieved by one-hot encoding the categories. Here the feature_names.csv file is also created which is used by the app.py. Next the training and testing split is made. Two model pipelines are made to test which model is the best on the data, and the models are tuned to the number of trees testing on 100, 200 and 300 trees. Then the models are trained on the X_train and y_train data. Afterwards the models are evaluated by testing them on the X_test and y_test data. The best model was the GradientBoostedClassifier, 'gb', so that is the model that is saved to a .pkl file.


Then we move on to the app.py file. This file loads the gb-model.pkl file and initially creates the categorical options for the user, which is created by using the feature_columns file. Next the categorical options are sorted to match the model's structure. The first and only app.route is created that receives user inputs and outputs the prediction and a history table. 
The first past of the POST request receives the categorical and numerical inputs while also handling any wrongdoings of the user. The categorical and numerical values are also loaded into lists for future handling. 
X loads the first part of the features into a list, the numericals. The categorical features have to be one_hot encoded and this is done by the function: one_hot_encoding_bool. It takes categorical and boolean answers and transforms them into 1's and 0's and also appends it to the feature list, X. Then the X-list is transformed into a 1D array that the model.predict_proba() function can load and a prediction is made. Depending on the prediction a different outcome will be output to the user. All the input data from the user is then uploaded to the database and a history query is sent to the database. 
A lot of data is returned to the user and the html. The html needs to know the different options for each category, so every category is sent. The results are sent and the history query is also sent. 


The layout.html file serves as the layout of the webpage and most importantly loads bootstrap and displays error messages to the user. 
If the user forgets to input any categorical or numerical values an alert will pop-up with a message explaining the issue. To display the alerts i have used flash and javascript. The different messages are implemented into the app.py file and if they are activated they will activate an eventlistener in the layout.html file. This eventlistener will then forward the message and activate an alert box that the user can then close. 

Moving on, the predict.html file is only focused on receiving inputs from the user while also showing the right options for each categories and the right values for the history database. 