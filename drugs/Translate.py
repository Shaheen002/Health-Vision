import langid
from googletrans import Translator
translator = Translator()
class Translate:
    def translate_to_english(self,text):
        translation = translator.translate(text, src='auto', dest='en')
        print(translation.text)
        return translation.text 
    # Function to translate text to the original language
    def translate_to_original_language(self,text, original_lang):
        translation = translator.translate(text, src='en', dest=original_lang)
        return translation.text
    def translate_to_hindi(self,text):
     translated_text = ''
     chunk_size = 500  # Adjust the chunk size as needed
     for i in range(0, len(text), chunk_size):
         chunk = text[i:i+chunk_size]
         translation = translator.translate(chunk, dest='hi').text
         translated_text += translation + ' '
     translated_text = translated_text.strip()
     return translated_text