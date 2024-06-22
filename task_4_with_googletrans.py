import speech_recognition as sr
from googletrans import Translator
from datetime import datetime
import pytz

def get_current_time():
    timezone = pytz.timezone('Asia/Kolkata')
    return datetime.now(timezone)

def translate_to_hindi(word):
    if word.startswith(('M', 'O', 'm', 'o')):
        print("Cannot translate words startting with M, O, m, o.")
    else:
        translator = Translator()
        translated_word = translator.translate(word, src='en', dest='hi').text
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