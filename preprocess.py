import os
import glob
import time
import math

import torch
import torchaudio
import torchaudio.transforms as T
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sg
import soundfile as sf

# Parameter for changing waveform to spectrogram
n_fft = 512


# Save as numpy file (recommend)
def getSTFT( directory, down_sample=16000, num_frames=16, n_fft=512 ):
    create_dir = os.getcwd() + "/numpy_data/"
    spec_frames = list()
    angles = list()

    #create directory
    if not os.path.exists( create_dir ):
        # If the dircetry does not exist, we create it.
        os.makedirs( create_dir )

    if directory == "input":
        path = os.path.join(".", directory, "*.wav")
    else:
        path = os.path.join(".", "dataset", directory, "*.wav")

    files = glob.glob( path )
    total = len( files )

    for i, filepath in enumerate( files ):
        # =================== Waveform preprocessing ===================
        waveform, sample_rate = sf.read( filepath )

        # downsampling 48kHz -> 16kHz
        down_sampled_waveform = sg.resample_poly( waveform, down_sample, sample_rate )
        # ===============================================================

        # =================== STFT ===================
        _, _, spec =sg.stft( down_sampled_waveform, fs=down_sample, window='hann', nperseg=n_fft, noverlap=n_fft//2 )
        # Remove the highest height value
        spec = spec[ :-1 ]
        angle = np.angle( spec )
        with np.errstate( divide='ignore' ):
            spec = np.log10( np.abs( spec ) )
        # ============================================

        # Divided into 16 frames
        num_seg = math.floor( spec.shape[1] / num_frames )
        for j in range( num_seg ):
            spec_frames.append( spec[ :, int( j * num_frames ) : int( ( j + 1 ) * num_frames ) ] )
            angles.append( angle[ :, int( j * num_frames ) : int( ( j + 1 ) * num_frames ) ] )
        
        print( "\rProgress : {:<5}/ {:<5} | {:.1f} %".format( i+1, total, 100 * ( float(i+1) / float(total) ) ), end='' )

    print()

    spec_frames = np.array( spec_frames )
    angles = np.array( angles )
    return spec_frames, angles