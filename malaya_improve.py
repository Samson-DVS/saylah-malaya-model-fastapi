import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from fastapi import FastAPI, HTTPException, Response, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import malaya_speech
import io
import soundfile as sf
from dotenv import load_dotenv

# Load environment variables
load_dotenv(".env")
app = FastAPI()

# API key security
API_KEY = os.getenv("AUTHKEY")
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

class TextToSpeechRequest(BaseModel):
    text: str

# Load models at startup
fs2 = malaya_speech.tts.fastspeech2(model='female-singlish')
vocoder = malaya_speech.vocoder.melgan()

async def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key

@app.post("/text-to-speech")
async def text_to_speech(request: TextToSpeechRequest, api_key: str = Depends(get_api_key)):
    try:
        # Generate speech from text
        r_singlish = fs2.predict(request.text)
        
        # Extract mel spectrogram
        mel_spectrogram = r_singlish['mel-output']
        
        # Generate audio from mel spectrogram
        y_ = vocoder(mel_spectrogram)
        audio = malaya_speech.astype.float_to_int(y_)

        # Convert audio to bytes
        buffer = io.BytesIO()
        sf.write(buffer, audio, 22050, format='wav')
        audio_bytes = buffer.getvalue()

        return Response(content=audio_bytes, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
