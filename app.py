from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
import malaya_speech
import io
import soundfile as sf
import uvicorn
import asyncio
import nest_asyncio
import numpy as np

app = FastAPI()

class TextToSpeechRequest(BaseModel):
    text: str

# Load the TTS model at startup
fs2 = malaya_speech.tts.fastspeech2(model='female-singlish')
vocoder = malaya_speech.vocoder.melgan()

@app.post("/text-to-speech")
async def text_to_speech(request: TextToSpeechRequest):
    try:
        # Generate speech from text
        r_singlish = fs2.predict(request.text)
        
        print("Type of r_singlish:", type(r_singlish))
        if isinstance(r_singlish, dict):
            print("Keys in r_singlish:", r_singlish.keys())
            for key, value in r_singlish.items():
                print(f"Shape of {key}:", np.shape(value) if isinstance(value, np.ndarray) else "Not a numpy array")
        
        # Use 'mel-output' key for the mel spectrogram
        if 'mel-output' in r_singlish:
            mel_spectrogram = r_singlish['mel-output']
        else:
            raise ValueError("Expected 'mel-output' key not found in FastSpeech2 output")

        # Ensure mel_spectrogram is a 2D numpy array with shape (time_steps, 80)
        while len(mel_spectrogram.shape) > 2:
            mel_spectrogram = np.squeeze(mel_spectrogram, axis=0)
        
        print("Final shape of mel_spectrogram:", mel_spectrogram.shape)

        # Pass the mel spectrogram to the vocoder
        y_ = vocoder(mel_spectrogram)
        audio = malaya_speech.astype.float_to_int(y_)

        # Convert audio to bytes
        buffer = io.BytesIO()
        sf.write(buffer, audio, 22050, format='wav')
        audio_bytes = buffer.getvalue()

        return Response(content=audio_bytes, media_type="audio/wav")
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}\n\nTraceback:\n{error_details}")

def run_server():
    nest_asyncio.apply()
    asyncio.get_event_loop().run_until_complete(
        uvicorn.run(app, host="0.0.0.0", port=8000)
    )

if __name__ == "__main__":
    run_server()
