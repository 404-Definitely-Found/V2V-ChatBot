# brew install portaudio
# pip install "assemblyai[extras]"
# pip install elevenlabs==0.3.0b0
# brew install mpv
# pip install --upgrade openai

# Load model directly
# Use a pipeline as a high-level helper
from huggingface_hub import login
login(token="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")


from transformers import pipeline

from transformers import Conversation, ConversationalPipeline



# import assemblyai as aai
# from elevenlabs import generate, stream
# from openai import OpenAI

from RealtimeSTT import RealtimeSTT

class AI_Assistant:
    def __init__(self):
        self.transcriber = None
        self.full_transcript = [
            {"role": "system", "content": "You are a receptionist at a dental clinic. Be resourceful and efficient."},
        ]

    def start_transcription(self):
        self.transcriber = RealtimeSTT.RealtimeTranscriber(
            sample_rate=16000,
            on_data=self.on_data,
            on_error=self.on_error,
            on_open=self.on_open,
            on_close=self.on_close,
            end_utterance_silence_threshold=1000
        )

        self.transcriber.connect()
        microphone_stream = RealtimeSTT.extras.MicrophoneStream(sample_rate=16000)
        self.transcriber.stream(microphone_stream)

    def stop_transcription(self):
        if self.transcriber:
            self.transcriber.close()
            self.transcriber = None

    def on_open(self, session_opened):
        # Handle session opened event
        pass

    def on_data(self, transcript):
        if not transcript.text:
            return

        if isinstance(transcript, RealtimeSTT.RealtimeFinalTranscript):
            self.generate_ai_response(transcript)
        else:
            print(transcript.text, end="\r")

    def on_error(self, error):
        # Handle errors
        pass

    def on_close(self):
        # Handle session closed event
        pass


###### Step 3: Pass real-time transcript to Llama3.0 ######



    def generate_ai_response(self, transcript):
        # Stop the transcription
        self.stop_transcription()

        # Append user input to full transcript
        self.full_transcript.append({"role": "user", "content": transcript.text})
        print(f"\nPatient: {transcript.text}", end="\r\n")

        # Tokenize the input transcript
        inputs = self.tokenizer.encode(transcript.text, return_tensors="pt")

        # Generate AI response
        with torch.no_grad():
            output = self.model.generate(inputs, max_length=50, num_return_sequences=1)

        # Decode the generated response
        ai_response = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Generate audio for the AI response
        self.generate_audio(ai_response)

        # Restart the transcription
        self.start_transcription()
        print(f"\nReal-time transcription: ", end="\r\n")





###### Step 4: Generate audio ######

def generate_audio(self, text):
        self.full_transcript.append({"role": "assistant", "content": text})
        print(f"\nAI Receptionist: {text}")
        # Use Realtime TTS to generate speech from text
        self.tts.speak(text)
        # stream(audio_stream)
        


greeting = "Hey, how may I assist you?"
ai_assistant = AI_Assistant()
ai_assistant.generate_audio(greeting)
ai_assistant.start_transcription()
