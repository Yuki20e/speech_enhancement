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

# Save as image (not used)
def wav2spectrogram( dir : str ):
    # Create a new directory
    cdir = os.getcwd()
    new_dir = cdir + "/spectrogram/img/" + dir + "_spec"
    if not os.path.exists( new_dir ):
        os.makedirs( new_dir )
    
    # Convert wave to spectrogram
    dataset_path = cdir + "/dataset/" + dir + "/*.wav"

    for i, filepath in enumerate( glob.glob( dataset_path ) ):
        # Sample rate is expected 48kHz
        waveform, sample_rate = torchaudio.load( filepath )

        # Downsampling 48kHz => 16kHz 
        down_sampled_waveform = torchaudio.functional.resample( waveform, sample_rate, sample_rate // 3)

        # Spectrogram (STFT)
        spectrogram = T.Spectrogram( n_fft=n_fft )
        spec = spectrogram( down_sampled_waveform )

        # Adjust the spectrogram
        spec_data = librosa.power_to_db( torch.flipud( spec[ 0 ] ) )
        spec_data = np.delete( spec_data, 0, axis=0 )

        filename = os.path.splitext( os.path.basename( filepath ) )[0]

        plt.imsave( new_dir + "/" + filename + ".png", spec_data )
        plt.close()

        print( "\r{:.1f} %".format( i / len( glob.glob( dataset_path ) ) * 100 ), end="" )


# Save as numpy file (recommend)
def getSTFT( directory, down_sample=16000, num_frames=16, n_fft=512 ):
    create_dir = "./spectrogram/numpy_data/"
    spec_frames = list()
    angles = list()

    #create directory
    if not os.path.exists(create_dir):
        # If the dircetry does not exist, we create it.
        os.makedirs(create_dir)

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