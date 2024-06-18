import os
import time

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

class ASR:
    def __init__(self, model_id='openai/whisper-large-v3'):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.pipe = pipeline(
            'automatic-speech-recognition',
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=128,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
    
    def simple_call(self, audio_path):
        result = self.pipe(audio_path)
        return result['text']

ASR_system = ASR(model_id='openai/whisper-large-v3')

t0 = time.time()
result = ASR_system.simple_call('data/test/Chua_bao_gio.mp3')
t1 = time.time()

print('Result: ', result)
print('Time cost: ', t1-t0)
