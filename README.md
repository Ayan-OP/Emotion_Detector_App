## Introduction About the Data :
***

The dataset: The goal is to predict **_emotion_** of the input text. (NLP & Deep Learning Problem)

The dataset contains:
* **_comment_** : Random Emotions text
* **_emotion_** : Emotions Category

Target :
* **_emotion_** : Emotion of the input text

Dataset: train.txt

#### It is observed that the dataset include 6 categories : anger, joy, sadness, fear, surprise & love.  

## Screenshot of UI
***

SS Link: https://drive.google.com/file/d/1488NOs8k_bvbXyM6-_e_s_kYRCHRSYx3/view




## Approach for the project
***

1. ### Loading data & libraries :
* Imported important libraries related to NLP and Deep Learning techniques.
* The data is first read as csv.
* Then the data is split into training, validating and testing data and saved as csv file.

2. ### Exploratory Data Analysis :
* We removed the duplicate values.
* We compared the data distributions of different categories of emotions of input text.
* We identified the word cloud for each emotion using the wordcloud library.

2. ### Data Cleaning & Preprocessing :
* Using Label Encoder we encoded the emotions categories into numerical form.
* We cleaned the input text to remove stop words, applied stemming using PorterStemmer and finally vectorized the text using TF-IDF vectorizer. All these things are necessary for the model to compute results.

3. ### Model Training :
* We first used machine learning algorithms to get a rough idea about the performance of our dataset. 
* We used Multinomial Naive Bayes, Logistic regression, Random Forrest, support Vector machine. Among these Random Forest model was found to be performing the best giving a 85% accuracy in predictions.
* Finally we used the LSTM model where we applied techniques such as padding, embedding, dropout, earlystopping etc. to build our final model and it gave us an accuracy of 
* This LSTM model is saved as h5 file.

4. ### Prediction Pipeline :
* We used random text as input, preprocessed it and then passed it to the model and were able to successfully categorize the text based on the 6 emotion categories.

5. ### Streamlit App creation :
* Streamlit app is created with User Interface where we used the trained LSTM model to predict the emotion of the input text inside a Web Application.

6. ### Training Notebook
* [Jupyter Notebook](https://github.com/Ayan-OP/Diamond_Price_Predictor_Project/blob/main/notebooks/EDA.ipynb)# Emotion_Detector_App
