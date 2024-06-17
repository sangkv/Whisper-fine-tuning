import os
import time

import torch
import whisper

class ASR:
    def __init__(self, model_name='large-v3'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = whisper.load_model(model_name).to(self.device)
        self.is_multilingual = self.model.is_multilingual
    
    def simple_call(self, audio_path, language=None):
        if self.is_multilingual:
            result = self.model.transcribe(audio_path, language=language)
        else:
            result = self.model.transcribe(audio_path, language='en')
        return result['text']
    
    def advanced_call(self, audio_path):
        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
        
        # detect the spoken language
        _, probs = self.model.detect_language(mel)
        # print(f"Detected language: {max(probs, key=probs.get)}")

        # decode the audio
        options = whisper.DecodingOptions()
        result = whisper.decode(self.model, mel, options)

        # the recognized text
        return result.text

ASR_system = ASR(model_name='large-v3')

t0 = time.time()
result = ASR_system.simple_call('data/test/2020.ogg', language='vi')
t1 = time.time()

print('Result: ', result)
print('Time cost: ', t1-t0)

