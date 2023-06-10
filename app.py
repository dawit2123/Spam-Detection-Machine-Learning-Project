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
    st.text("💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥")
    st.title("Normal or Spam mail detector made by Group 7")
    st.text("💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥💥")
    image = Image.open('spam.jpeg')

    st.image(image)

    st.subheader("Enter an email to check it as spam or normal")

    # Input form for the user to enter email text
    input_mail = st.text_area("Email Text", "")
    if st.button("Submit",):
        # Convert input text to feature vectors
        input_data_features = feature_extraction.transform([input_mail])

        # Make prediction
        prediction = model.predict(input_data_features)
        if prediction[0] == 1:
            st.subheader("The above mail is : Normal mail")
        else:
            st.subheader("The above mail is : Spam mail")

if __name__ == '__main__':
    main()

st.subheader("Contributers of this Machine Learning project")
st.text("============================")
st.text("Name                  ID")
st.text("============================")
st.text("Dawit Zewdu Munie 1307571")
st.text("Ephrem Habtamu    1308250")
st.text("Fentahun Mengie   1306919")
st.text("Jemal Workie      1307712")
st.text("Solomon Muhye     1309375")
