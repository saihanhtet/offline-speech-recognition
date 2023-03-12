import whisper
import pyaudio
import wave
import audioop

def record_wav(filepath="audio/output.wav"):
    start = False
    slience_frame = 0
    THRESHOLD = 50
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1,
                        rate=44100, input=True, frames_per_buffer=1024)
    frames = []
    try:
        print('Recording...')
        while True:
            data = stream.read(1024)
            frames.append(data)
            rms = audioop.rms(data, 2)
            if not start and rms > THRESHOLD:
                start = True
            if start and rms < THRESHOLD:
                slience_frame += 1
            if rms > THRESHOLD:
                slience_frame = 0
            if rms < THRESHOLD and int(slience_frame) > 80:
                print('Recording stopped')
                break
    except KeyboardInterrupt:
        pass
    stream.stop_stream()
    stream.close()
    audio.terminate()

    sound_file = wave.open(filepath, 'wb')
    sound_file.setnchannels(1)
    sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    sound_file.setframerate(44100)
    sound_file.writeframes(b''.join(frames))
    sound_file.close()
    print('Finished recording.')
    return filepath


def recognize(audiofile=None, data=False):
    model = whisper.load_model("base")
    beam_size = 5
    best_of = 5
    temperature = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    decode_options = dict(language="en", best_of=best_of,
                          beam_size=beam_size, temperature=temperature, fp16=False)
    transcribe_options = dict(task="transcribe", **decode_options)
    if audiofile:
        result = model.transcribe(audiofile, **transcribe_options)
    if data:
        return result
    else:
        return result['text'].strip()


file = record_wav()
res = recognize(audiofile=file)
print(res)
