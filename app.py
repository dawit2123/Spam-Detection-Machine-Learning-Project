import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image


# Loading the data from csv file to a pandas DataFrame
raw_mail_data = pd.read_csv('mail_data.csv')

# Replace the null values with a null string
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')

# Label spam mail as 0 and ham mail as 1
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1

# Separating the data as texts and labels
X = mail_data['Message']
Y = mail_data['Category']

# Transform the text data to feature vectors
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_features = feature_extraction.fit_transform(X)

# Convert Y values to integers
Y = Y.astype('int')

# Splitting the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_features, Y, test_size=0.2, random_state=3)

# Training the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, Y_train)
# Define the Streamlit app
def main():
    st.text("💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥")
    st.title("Normal or Spam email detector made by Group 7")
    st.text("💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥")
    image = Image.open('spam.jpeg')

    st.image(image)

    st.subheader("Enter an email to check it as spam or normal👇👇👇👇👇")

    # Input form for the user to enter email text
    input_mail = st.text_area("Email Text", "")
    if st.button("Submit",):
        # Convert input text to feature vectors
        input_data_features = feature_extraction.transform([input_mail])

        # Make prediction
        prediction = model.predict(input_data_features)
        if prediction[0] == 1:
            st.text("👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇")
            st.subheader("The above email is : Normal email😊😊😊😊😊😊")
            st.text("☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️")
        else:
            st.text("👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇")
            st.subheader("The above email is : Spam email😭😭😭😭😭😭😭")
            st.text("☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️")

if __name__ == '__main__':
    main()
st.text("👉👉INSTRUCTIONS : This machine learning project is made by Group 7 for educational\n purposes only, and It is fed with a limited amount of Data to train it. You can \n check, for example, by writing the down examples in the text area.\n 👉👉 NORMAIL EMAIL EXAMPLE: Your assignment has been accepted by the teacher, and \n you can check your results by June 23 , 2023.\n 👉👉 SPAM EMAIL EXAMPLE: Congratulations! You've been selected to receive a free \n vacation to Langano Resort, Hawassa. Click here to claim your prize!")
st.text("                         ")
st.text("                         ")
st.text("                         ")
st.subheader("Contributers of this Machine Learning project")
st.text("Dawit Zewdu")
st.text("Copyright © 2023 Dawit Zewdu Munie")
