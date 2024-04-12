import streamlit as st
import re
from nltk.stem import PorterStemmer
import pickle
import nltk
from keras.models import load_model
from keras.preprocessing.text import one_hot
from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
import numpy as np
import string
from PIL import Image


# =========================repeating the same functions==========================================
## Perfoming the text preprocessing

nltk.download('stopwords')  # Downloading all the stopwords from the nltk library
pattern = re.compile('<.*?>')  # Pattern for removing the HTML tags
punctuation = string.punctuation   # Extracting all punctuation from the string library
ps = PorterStemmer()  # Creating a PorterStemmer object for the stemming purpose
tokenizer = Tokenizer() # Creating a Tokenizer object for representing the text into numeric form

def text_preprocess(text):

  text = re.sub(pattern,'',text)  # Removing the HTML tags using re library

  text = text.lower()  # Lower case all the character present in the text

  text = text.translate(str.maketrans('','',punctuation))   # Removing all the punctuation from the text

  text = text.split()    # word tokenize the text

  text = [ps.stem(word) for word in text if word not in stopwords.words('english')]  # Removing the stopwords from the text and stem each word

  return ' '.join(text)  # Join each word for the formation of clear text in string form


#========================loading the save files==================================================
lb = pickle.load(open('label_encoder.pkl','rb'))

# Function to load LSTM model
def load_modelop():
    model = load_model('C:\\Users\\ayaan\\Downloads\\Emotion_Detector\\model1.h5', compile=False)
    return model

model = load_modelop()



def predict_emotion_of_text(text):
    
    processed_text = text_preprocess(text)
    text_to_sequence = tokenizer.texts_to_sequences([processed_text])[0]
    padded_sequence = pad_sequences([text_to_sequence],maxlen = 50, padding = 'post')
    
    prediction = model.predict(padded_sequence)[0]   
    
    classes = ['sadness','joy','love','anger','fear','surprise']

    return classes[np.argmax(prediction)], prediction

#==================================creating app====================================
# App
# Set custom CSS styles

st.markdown(
    """
    <style>
    body {
        font-family: 'Poppins', sans-serif;
        background-color: #f0f2f5;
    }
    .title {
        font-size: 48px;
        text-align: center;
        color: #333333;
        margin-top: 30px;
        margin-bottom: 30px;
    }
    .header {
        font-size: 24px;
        font-weight: bold;
        color: #333333;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .text-input {
        font-size: 18px;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #cccccc;
        width: 400px;
        margin-top: 10px;
    }
    .button {
        font-size: 18px;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 5px;
        background-color: #0074D9;
        color: #ffffff;
        margin-top: 10px;
    }
    .result {
        font-size: 20px;
        margin-top: 20px;
        margin-bottom: 10px;
        color: #333333;
    }
    .emotion-image {
        margin-top: 20px;
        margin-bottom: 30px;
        width: 300px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def main():
    # Set the title and description of the app
    st.markdown("<h1 class='title'>Human Emotions Detection App</h1>", unsafe_allow_html=True)
    st.markdown("---")
    st.write("Enter your text and press 'Predict' to detect emotions.")

    # Create a text input box for the user to enter their text
    user_input = st.text_input("Enter your text:", "", key="text_input")

    # Create a button to trigger emotion detection
    if st.button("Predict", key="predict_button"):
        # Perform emotion detection on the user input
        emotion, result = predict_emotion_of_text(user_input)
        st.write("Predicted Emotion:", emotion)
        st.write("Probability:", np.max(result))


        # # Display an image based on the detected emotion
        # if result[0][0]> 0.5:  # If Anger score is above 0.5
        #     image = Image.open("C:\\Users\\ayaan\\Downloads\\Emotion_Detector\\angry.png")
        #     st.image(image, caption="Anger", use_column_width=True, output_format='PNG', 
        #              width=300)
        # elif result[0][1]> 0.5:  # If Fear score is above 0.5
        #     image = Image.open("C:\\Users\\ayaan\\Downloads\\Emotion_Detector\\fear.png")
        #     st.image(image, caption="Fear", use_column_width=True, output_format='PNG', 
        #              width=300)
        # elif result[0][2] > 0.5:  # If Joy score is above 0.5
        #     image = Image.open("C:\\Users\\ayaan\\Downloads\\Emotion_Detector\\joy.png")
        #     st.image(image, caption="Joy", use_column_width=True, output_format='PNG', 
        #              width=300)
        # elif result[0][3] > 0.5:  # If Love score is above 0.5
        #     image = Image.open("C:\\Users\\ayaan\\Downloads\\Emotion_Detector\\love.png")
        #     st.image(image, caption="Love", use_column_width=True, output_format='PNG', 
        #              width=300)
        # elif result[0][4] > 0.5:  # If Sadness score is above 0.5
        #     image = Image.open("C:\\Users\\ayaan\\Downloads\\Emotion_Detector\\sadness.png")
        #     st.image(image, caption="Sadness", use_column_width=True, output_format='PNG', 
        #              width=300)
        # elif result[0][5] > 0.5:  # If Surprise score is above 0.5
        #     image = Image.open("C:\\Users\\ayaan\\Downloads\\Emotion_Detector\\surprise.png")
        #     st.image(image, caption="Surprise", use_column_width=True, output_format='PNG', 
        #              width=300)

if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

        


        
        
        
    











    
