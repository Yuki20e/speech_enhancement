import preprocess
from train import UNet, training
import eval

import torch
import numpy as np
import sys
import os
from scipy import signal as sg
from scipy.io import wavfile
from matplotlib import pyplot as plt

numpy_data_path = os.getcwd() + "/numpy_data/"
down_sample = 16000
num_frames = 16
lr = 0.0001
batch_size = 64
n_fft = 512
sample_rate = 16000

frame_length = 20

preprocessing_flag = False
training_flag = False

def spec2wave( spec, angles, sample_rate=16000, n_fft=512 ):
    spec = np.power( 10, spec )
    spec = spec * np.exp( 1j * angles )
    spec = np.append( spec, spec[-1, :][np.newaxis,:], axis=0 )
    _, wave = sg.istft( spec, fs=sample_rate, window='hann', nperseg=n_fft, noverlap=n_fft//2 )
    return wave


def main():
    args = sys.argv
    device = torch.device( args[ 1 ] )
    print( "device : {} ".format( device )  )

    ### Preprocessing ###
    if preprocessing_flag:
        # Noisy audio
        train_X, _ = preprocess.getSTFT( "noisy_trainset_28spk_wav", down_sample, num_frames, n_fft )

        # Clean audio
        train_Y, _ = preprocess.getSTFT( "clean_trainset_28spk_wav", down_sample, num_frames, n_fft )

        # Save as numpy file .npy
        np.save( numpy_data_path + "noisy_train", train_X )
        np.save( numpy_data_path + "clean_train", train_Y )
    else:
        # Load spectrogram data file from .npy
        train_X = np.load( numpy_data_path + "noisy_train.npy" )
        train_Y = np.load( numpy_data_path + "clean_train.npy" )

    # Normalization
    max_x = np.amax( train_X )
    min_x = np.amin( train_X )
    print( f"max_x : {max_x} min_x : {min_x}")
    train_X = ( train_X - min_x ) / ( max_x - min_x )
    train_Y = ( train_Y - min_x ) / ( max_x - min_x )

    # Split data
    assert( len( train_X ) == len( train_Y ) )
    train_X = torch.tensor( train_X, dtype=torch.float32 )
    train_Y = torch.tensor( train_Y, dtype=torch.float32 )

    ### Training ###
    if not os.path.exists( os.getcwd() + "/trained_model" ):
        # If the dircetry does not exist, we create it.
        os.makedirs( os.getcwd() + "/trained_model" )
    # model_name = os.getcwd() + "/trained_model/model-30-" + args[ 1 ] +".pth"
    model_name_cpu = os.getcwd() + "/trained_model/model-30-cpu.pth"
    model = UNet().to( device )
    if training_flag:
        model, loss = training( model, device, lr, train_X, train_Y, 30, batch_size )
        model_cpu = model.to( 'cpu' )
        # torch.save( model.state_dict(), model_name )
        torch.save( model_cpu.state_dict(), model_name_cpu )
    else:
        model.load_state_dict( torch.load( "./trained_model/model-30-cpu.pth", map_location=torch.device( device ) ) )

    ### Main process ###
    input = args[ 2 ]
    input_audio, angles = preprocess.getSTFT( input, sample_rate, num_frames, n_fft )
    input_audio = ( input_audio - min_x ) / ( max_x - min_x )
    input_audio = torch.from_numpy( input_audio.astype( np.float32 ) ).clone().detach()
    input_audio.to( device )
    output = eval.denoise( model, device, input_audio, max_x, min_x )

    output_specs = eval.combineFrames( output, frame_length )
    input_specs = eval.combineFrames( input_audio, frame_length )
    combined_angles = eval.combineFrames( angles, frame_length )

    if not os.path.exists( os.getcwd() + "/output" ):
    # If the dircetry does not exist, we create it.
        os.makedirs( os.getcwd() + "/output" )

    for i in range( output_specs.shape[ 0 ] ):
        fig = plt.figure( figsize=( 12, 5 ) )
        fig.suptitle( "Noised spectrogram (left), denoised (right)" )

        ax0 = fig.add_subplot( 1, 2, 1 )
        ax0.imshow( input_specs[ i ] )
        ax0.invert_yaxis()

        ax1 = fig.add_subplot( 1, 2, 2 )
        ax1.imshow( output_specs[ i ] )
        ax1.invert_yaxis()

        fig.savefig( "./output/out_" + str( i ) + ".png" )

    # wav出力実装予定
    # output_wave = list()
    # for i in range( output_specs.shape[ 0 ] ):
    #     output_wave.append( spec2wave( output_specs[ i ], combined_angles[ i ] ) )
    #     # soundfile.write( file='./output/out-' + str(i) + ".wav", data=output_wave[ i ], samplerate=sample_rate )
    #     wavfile.write( './output/out-' + str(i) + ".wav", sample_rate, output_wave[ i ] )


    

if __name__ == "__main__":
    main()