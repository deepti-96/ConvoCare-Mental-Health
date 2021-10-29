import re
import string
import numpy as np
from tensorflow import keras
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from constants import CLASSES

model = keras.models.load_model('./models/hDModel.h5')

def clean_text(text):
    text = text.lower()
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('\[[^]]*\]', ' ', text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    tokenizer = TreebankWordTokenizer()
    comment_tokens = tokenizer.tokenize(text)
    return comment_tokens

def remove_stopwords(text):
    stemmer = PorterStemmer()
    words = []

    for word in text:
        if(word not in stopwords.words('english')):
            stem_word = stemmer.stem(word)
            words.append(stem_word)
    return words

def output_prediction(text, max_features = 22000, maxlen = 200):
    tokenizer = Tokenizer(num_words = max_features)
    tokenizer.fit_on_texts(text)
    tokenized_train = tokenizer.texts_to_sequences(text)
    x_train = pad_sequences(tokenized_train, maxlen = maxlen)
    if len(x_train) == 0:
        return {
            "error" : "Not Found"
        }
    prediction = model.predict(x_train)
    prediction = np.sum(prediction, axis=0)
    result = []
    for i, value in enumerate(prediction):
        if i == 0:
            continue
        if value > 0.05:
            result.append(CLASSES[i])
    return {"result" : result if len(result) != 0 else [CLASSES[0]]}

def text_analysis(text):
    cleaned_text_data = clean_text(text)
    preprocessed_data = remove_stopwords(cleaned_text_data)
    return output_prediction(preprocessed_data)

if __name__ == "__main__":
    text = "I am sad. I am lonely. I need help."
    print(text_analysis(text))
