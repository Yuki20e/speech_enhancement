import numpy as np
from scipy import signal as sg

def spec2wave( spec, angles, sample_rate=16000, n_fft=512 ):
    spec = np.power( 10, spec )
    spec = spec * np.exp( 1j * angles )
    spec = np.append( spec, spec[ -1, : ][ np.newaxis,: ], axis=0 )
    _, wave = sg.istft( spec, fs=sample_rate, window='hann', nperseg=n_fft, noverlap=n_fft//2 )

    return wave

def combineFrames( frames, n=1 ):
    specs = list()
    for i in range( frames.shape[ 0 ] ):
        if i % n == 0:
            spec = frames[ i ]
        else:
            spec = np.concatenate( [ spec, frames[ i ] ], 1 )
        
        if i % n == n - 1:
            specs.append( spec )
        print( "\rProgress : {:<5}/ {:<5} | {:.1f} %".format( i+1, frames.shape[ 0 ], 100 * ( float( i+1 ) / float( frames.shape[ 0 ] )  ) ), end='' )
    print()

    specs = np.array( specs )
    
    return specs

def denoise( model, device, audio, max_x, min_x ):
    model.eval()
    outputs = list()
    
    audio = audio.unsqueeze( dim=1 )
    for i in range( audio.shape[ 0 ] ):
        x = audio[ i ].to( device )
        output = model( x )
        output = output * ( max_x - min_x ) - min_x

        if device == 'cpu':
            output = output.detach().numpy()
        else:
            output = output.cpu().detach().numpy()

        outputs.append( output )

        print( "\rProgress : {:<5}/ {:<5} | {:.1f} %".format( i+1, audio.shape[ 0 ], 100 * ( float( i+1 ) / float( audio.shape[ 0 ])  ) ), end='' )

    outputs = np.array( outputs )

    return outputs