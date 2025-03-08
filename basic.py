import streamlit as st
import pandas as pd
import numpy as np
from scipy.io import wavfile

sample_rate, audio_data = wavfile.read("rec.wav")
audio_data = audio_data.T
print(audio_data.shape)
print(sample_rate)
st.audio(audio_data, loop=True, sample_rate=sample_rate)

data = st.audio_input("record")
