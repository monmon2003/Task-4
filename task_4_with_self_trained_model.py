import speech_recognition as sr
from datetime import datetime
import pytz
import json
from keras.models import load_model
from keras.preprocessing.text import tokenizer_from_json
from keras.utils import pad_sequences
import numpy as np


model_2 = load_model('english_to_hindi_model')


with open('english_tokenizer.json') as f:
    data = json.load(f)
    english_tokenizer = tokenizer_from_json(data)
    


with open('hindi_tokenizer.json') as f:
    data = json.load(f)
    hindi_tokenizer = tokenizer_from_json(data)
    


with open('sequence_length_hindi.json') as f:
    max_length_hn = json.load(f)
    
def pad(x, length=None):
    return pad_sequences(x, maxlen=length, padding='post')

def translate(english_sentence):
    english_sentence = english_sentence.lower()
    
    english_sentence = english_sentence.replace(".", '')
    english_sentence = english_sentence.replace("?", '')
    english_sentence = english_sentence.replace("!", '')
    english_sentence = english_sentence.replace(",", '')
    
    english_sentence = english_tokenizer.texts_to_sequences([english_sentence])
    english_sentence = pad(english_sentence, max_length_hn)
    
    english_sentence = english_sentence.reshape((-1,max_length_hn))
    
    hindi_sentence = model_2.predict(english_sentence)[0]
    
    hindi_sentence = [np.argmax(word) for word in hindi_sentence]

    hindi_sentence = hindi_tokenizer.sequences_to_texts([hindi_sentence])[0]
    
    print("hindi translation: ", hindi_sentence)
    
    return hindi_sentence

def get_current_time():
    timezone = pytz.timezone('Asia/Kolkata')
    return datetime.now(timezone)

def translate_to_hindi(word):
    if word.startswith(('M', 'O', 'm', 'o')):
        print("Cannot translate words startting with M, O, m, o.")
    else:
        translated_word = translate(word)
        print(f"Translation in Hindi: {translated_word}")

def main():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    while True:
        current_time = get_current_time()
        if current_time.hour >= 18:
            print("Please try after 6 PM IST.")
            return
        
        print("Listening for an English word...")
        
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
        
        try:
            english_word = recognizer.recognize_google(audio)
            print(f"Recognized word: {english_word}")
            print("\n")
            translate_to_hindi(english_word)
        except sr.UnknownValueError:
            print("Could not understand the audio. Please repeat.")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")

if __name__ == "__main__":
    main()