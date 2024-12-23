import pyaudio
import wave

import numpy as np
import librosa

class AudioStream(object):

    def __init__(self, sr, fs, audio_path=None):

        self.audio_path = audio_path

        self.pa = pyaudio.PyAudio()
        self.wave_file = None

        self.fs = fs
        self.sr = sr
        if self.audio_path is None:
            # live audio input
            self.audio_stream = self.pa.open(format=pyaudio.paInt16, channels=1, rate=sr,
                                             input=True, frames_per_buffer=fs // 2)

        else:
            # read from file
            self.from_ = 0
            self.to_ = self.fs // 2
            self.wave_file, sr = librosa.load(audio_path)
            print(self.wave_file.shape)
            # self.wave_file = wave.open(self.audio_path, 'rb')
            self.audio_stream = self.pa.open(format=pyaudio.paFloat32,
                                             channels=1,
                                             rate=sr,
                                             output=True,)
            
            # print(self.wave_file.getsampwidth(), self.wave_file.getnchannels(), self.wave_file.getfram
            # erate(), self.wave_file.getnframes())
    
    
    def __len__(self) :
        if self.audio_path is None:
            return 0
        else:
            return len(self.wave_file)
        
    def get(self):

        if self.wave_file is None:
            data = self.audio_stream.read(self.fs // 2)
        else:
            data = self.wave_file[self.from_:self.to_]
            # print(self.from_, self.to_)
            self.from_ += self.fs // 2
            self.to_ += self.fs // 2
            # data = self.wave_file.readframes(self.fs // 4)
            
        if len(data) <= 0:
            data = None

        if data is not None:

            if self.wave_file is not None:
                # write data to audio stream
                self.audio_stream.write(data.astype(np.float32).tobytes())
            else:
                data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / (2 ** 15)

        return data
    
    # def get_sampleinfo(self):
        
    #     if self.wave_file is None:
    #         data = self.audio_stream.read(self.fs // 2)
    #     else:
    #         data = self.wave_file[self.from_:self.to_]
    #         # print(self.from_, self.to_)
    #         self.from_ += self.fs // 2
    #         self.to_ += self.fs // 2
    #         # data = self.wave_file.readframes(self.fs // 4)
            
    #     if len(data) <= 0:
    #         data = None

    #     if data is not None:

    #         if self.wave_file is not None:
    #             # write data to audio stream
    #             self.audio_stream.write(data.astype(np.float32).tobytes())
    #         else:
    #             data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / (2 ** 15)

    #     return data

    def close(self):
        if self.wave_file is not None and self.wave_file is None:
            self.wave_file.close()

        self.audio_stream.stop_stream()
        self.audio_stream.close()
        self.pa.terminate()
