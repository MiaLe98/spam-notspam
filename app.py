import streamlit as st
import os
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
import pickle
from pyrtable.record import BaseRecord
from pyrtable.fields import StringField, DateField, SingleSelectionField, \
        SingleRecordLinkField, MultipleRecordLinkField

def main(): 
  # Heading
  st.header("Spam or Not Spam")
  # Text area for user input
  user_input = st.text_area("Enter your text here")
  if st.button("Let's Go!"):
    df = pd.read_csv("spam-notspam-chat.csv")
    target = lambda row: (1 if row.category=='spam' else 0)
    df['target'] = df.apply(target, axis=1)
    #train test split
    X_train, X_test, y_train, y_test = train_test_split(df['phrase'], 
                                                        df['target'], 
                                                        random_state=0)
    vect = CountVectorizer().fit(X_train)
    text = [user_input]
    vector = vect.transform(text)
    if loaded_model.predict(vector.toarray()) == [0]: 
      st.markdown('**Not spam**')
    else: 
      st.markdown('**Spam**')
  
  options = list()
  radio = st.sidebar.selectbox("What's the correct option?",('Not spam', 'Spam'))
  if radio=='Not spam': 
      options.append('not spam')
  elif radio=='Spam': 
      options.append('spam')
  if st.sidebar.button("Send feedback!"): 
        new_record = SentimentAnalysisRecord(text=user_input, labels=options[0])
        new_record.save()

@st.cache(allow_output_mutation=True, show_spinner=False)
def get_model(): 
    loaded_model = pickle.load(open('spam-notspam.pkl', 'rb'))
    return loaded_model 

loaded_model = get_model()

class SentimentAnalysisRecord(BaseRecord):
    class Meta:
        # Open “Help > API documentation” in Airtable and search for a line
        # starting with “The ID of this base is XXX”.
        base_id = ''
        table_id = 'table-connected'

    @classmethod
    def get_api_key(cls):
        # The API key can be generated in you Airtable Account page. 
        # DO NOT COMMIT THIS STRING!
        return ''

    text = StringField('text')
    labels = StringField('labels')


if __name__ == "__main__":
    main()

