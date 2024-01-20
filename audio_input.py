import numpy as np
import multiprocessing as mp
import pyaudio
import time


def open_audio_stream(audio_queue, device_name_like=None):
    print(f'Finding speech using device_name_like={device_name_like}')
    device_idx = find_device_idx(device_name_like)
    print(f'Finding speech using device_idx={device_idx}')
    
    sample_rate = 16000
    chunk = 160 # 10ms
    channels = 1

    p = pyaudio.PyAudio()

    def callback(in_data, frame_count, time_info, status):
        audio_queue.put((time.time(), time_info, in_data))
        return (None, pyaudio.paContinue)

    print('Starting stream')

    stream = p.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk,
        input_device_index=device_idx,
        stream_callback=callback)

    while stream.is_active():
        time.sleep(0.1)

    print('Stopping stream')

    stream.close()
    p.terminate()


def find_device_idx(device_name_like=None):
    p = pyaudio.PyAudio()

    if not device_name_like:
        return p.get_default_input_device_info()['index']

    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if device_name_like in info['name']:
            print(f'Found device "{info["name"]}" with index {i}')
            return i

    return p.get_default_input_device_info()['index']


class AudioInput():

    def __init__(self, device_name_like=None):
        self.device_name_like = device_name_like
        self.audio_input_process = None

    def start(self):
        print('Starting')
        ctx = mp.get_context('spawn')
        self.audio_queue = ctx.Queue()
        self.audio_input_process = ctx.Process(
            target=open_audio_stream, args=(self.audio_queue, self.device_name_like, ))
        self.audio_input_process.start()

    def run(self, callback_fn):
        if self.audio_input_process is None:
            self.start()
            time.sleep(1)

        chunk_buffer = []

        while True:
            start_time = int(time.time() * 1000)

            while not self.audio_queue.empty():
                chunk_buffer.append(self.audio_queue.get())

            concatenated_bytes = b''.join(chunk[2] for chunk in chunk_buffer)
            callback_fn(concatenated_bytes)
            chunk_buffer = []

            # Sleep for enough time such that the loop is run every 1 second
            current_time = int(time.time() * 1000)
            sleep_time = max(0, (1000 - (current_time - start_time)) / 1000)
            print(f'Sleeping for {sleep_time} seconds')
            time.sleep(sleep_time)