# Neural_Network_Charity_Analysis

The report should contain the following:

# Overview of the analysis
Alphabet Soup has donated to over 34,000 organizations over the years, some of which were successful investments and others that were not. With the data provided we were asked to use machine learning techiques and apply neural network applications to the data and provide a predictive measure to determine which organizations would be safe deicisons and which are risky. 

This project explores the advantages and disadvantages of neural networks. This project is comprised of 3 steps:<br/>

- Preprocessing the data for the neural network<br/>
- Compile,train and evaluate the model<br/>
- Optimizing the model<br/>

In the pre-processing phase, I investigated the dataset and determining if any categories should be binned into an "other" category to reduce noise and balancing the neural network. I used OneHotEncoder to transform dataset into binary, split the preprocessed data into a training and testing dataset, and implemented a StandardScaler.<br/>

In the Compile, Train and Evaluate phase, several parameter were set to run the neural network using TensorFlow models such as setting the number of input features, number of hidden layers, number of units per node, and activation function. The model is then compiled with the "adam" optimizer, trained, and evulated by accuracy.<br/>

In the Optimization phase, contains 3 iterations of the original model with several perameters changed in hopes of reaching a 75% threshold. 

# Results

## Data Preprocessing
What variable(s) are considered the target(s) for your model?<br/>
Checking to see if the target is marked as IS_SUCCESSFUL in the DataFrame, indicating that it has been successfully funded by AlphabetSoup.<br/>

What variable(s) are considered to be the features for your model?<br/>
The following columns APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT are the features for our model.<br/>

What variable(s) are neither targets nor features, and should be removed from the input data?<br/>
The EIN and NAME columns will not increase the accuracy of the model and can be removed to improve code efficiency.<br/>

# Compiling, Training, and Evaluating the Model

How many neurons, layers, and activation functions did you select for your neural network model, and why?<br/>

### Orginal Parameters 

-First hidden layer<br/>
nn.add(tf.keras.layers.Dense(units=80, activation="sigmoid", input_dim = number_input_features))<br/>

-Second hidden layer<br/>
nn.add(tf.keras.layers.Dense(units=20, activation="relu"))<br/>

-Output layer<br/>
nn.add(tf.keras.layers.Dense(units=1, activation="linear"))<br/>
-epochs=5<br/>
-Accuracy: 0.7218658924102783<br/>

### Optimzation Trail # 1

-First hidden layer<br/>
nn.add(tf.keras.layers.Dense(units=90, activation="sigmoid", input_dim = number_input_features))<br/>

-Second hidden layer<br/>
nn.add(tf.keras.layers.Dense(units=30, activation="relu"))<br/>

-third hidden layer<br/>
nn.add(tf.keras.layers.Dense(units=20, activation="sigmoid"))<br/>

-Output layer<br/>
nn.add(tf.keras.layers.Dense(units=1, activation="linear"))<br/>

-ModelCheckpoint<br/>
-epochs=5<br/>
-Accuracy: 0.723498523235321<br/>

### Optimzation Trail # 2

-First hidden layer<br/>
nn.add(tf.keras.layers.Dense(units=80, activation="relu", input_dim = number_input_features))<br/>

-Second hidden layer<br/>
nn.add(tf.keras.layers.Dense(units=30, activation="relu"))<br/>

-Third hidden layer<br/>
nn.add(tf.keras.layers.Dense(units=10, activation="relu"))<br/>

-Output layer<br/>
nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))<br/>

-ModelCheckpoint<br/>
-period=5<br/>
-epochs=20<br/>
-Accuracy: 0.7258309125900269<br/>

### Optimzation Trail # 3

-First hidden layer<br/>
nn.add(tf.keras.layers.Dense(units=100, activation="relu", input_dim = number_input_features))<br/>

-Second hidden layer<br/>
nn.add(tf.keras.layers.Dense(units=30, activation="sigmoid"))<br/>

-Third layer<br/>
nn.add(tf.keras.layers.Dense(units=10, activation="sigmoid"))<br/>

-Output layer<br/>
nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))<br/>

-ModelCheckpoint<br/>
-period=5<br/>
-epochs=20<br/>
-Accuracy: 0.7272303104400635<br/>

Were you able to achieve the target model performance?<br/>

-No with each iteration the model became marginally better but no signficant changes.<br/>

What steps did you take to try and increase model performance?<br/>

-Additional neurons are added to hidden layers <br/>
-Additional hidden layers are added<br/>
-The activation function of hidden layers or output layers is changed for optimization<br/>
-The model's weights are saved every 5 epochs<br/>
-The results are saved to an HDF5 file<br/>


# Summary

Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and explain your recommendation.

The models accuracy started with 72.1% ans ended up being 72.7%. It was recommended to reduce noisy and irrelenvant data, however I had no ability to determine what data was neccessary and what was not. The best way to increase the accuracy of your model is to have more data. We could use a supervised machine learning model such as the Random Forest Classifier to generate a classified output and evaluate its performance against our deep learning model.