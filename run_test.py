from chatterbox.mtl_tts import (
    ChatterboxMultilingualTTS,
    SUPPORTED_LANGUAGES
)

import torch

original_torch_load = torch.load

def patched_torch_load(f, map_location=None, **kwargs):
    if map_location is None:
        map_location = 'cpu'
    return original_torch_load(f, map_location=map_location, **kwargs)

torch.load = patched_torch_load

if __name__ == "__main__":
    map_location=torch.device('cpu')
    try:
        MODEL = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
        torch.load = original_torch_load
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    kr_text = ' 밖에 비가 많이 오는데 집에 어떻게 가지?'
    kr_wav = MODEL.generate(kr_text, language_id='ko')
    print(len(kr_wav))