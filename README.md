# speech enhancement

## Introduction
Nowadays we often make a call or online meeting using several voice chat applications such as Zoom, Skype, Discord, and so on. It is convenient because we can contact wherever. However, the calls are easily affected by background noise due to their environment. Thus, we cannot hear the other voice clearly due to the noise because it can lower the quality of the call.  Denoising from sound wave technique can apply to call in noisy space or reduce error of voice detection.

Our objective is to implement U-Net convolutional neural network for denoising from audio file. We also examine this algorithm and find the impact by adjustment any parameters such as kernel size or strides.

We use the clean and noised voice dataset presented by Cassia Valentini-Botinhao et al [1]. This dataset including 30 speakers balanced between male and femal, and same accent audio voice about 5 seconds. We divided the noised voice dataset into train and test data which are 28 and 2 speakers respectively. The train dataset is created with 10 types of noise with signal-to-noise ratio (SNR) values of 15 dB, 10dB, 5dB, and 0 dB which is obtained by the Demand database [2]. The test dataset includes 5 types of noise are also selected from Demand database and SNR is 17.5dB, 12.5dB, 7.5dB and 2.5dB.

In implementation, we construct neural network based on study by Ahmet E Bulut et al [3] for denoising with a spectrogram image converted by sound wave as input. And loss is calculated by comparing clear audio data and trained noisy audio data. U-Net convolutional neural network encodes the input features with 8 2D convolution layers and decodes with 8 2D convolution layers. We employ the preprocessing of input audio file from [3].

There are other efficient audio denoising technique such as Wave-U-Net[4].  The implementation is complex compared to U-Net CNN since the method should receive wave data as input. On the other hand, U-Net CNN is input image data same as original U-Net algorithm. Thus, we can implement simply and reduce computation time to obtain the output.

This is the project created in the master's class with some modifications and functions added

## Usage
```
pip3 install -r requirements.txt
python3 main.py [cpu, mps, cuda]
```
引数は自分の環境に合わせて指定してください.
You can choose to use cpu, mps or cuda in training. 

## Dataset
モデルを作成したい場合はデータセットを下記のリンクからダウンロードし展開してdatasetディレクトリにおいてください.\
訓練データは, "clean_trainset_28spk_wav.zip"と"noisy_trainset_28spk_wav.zip "になります.\
If you want to create a AI model, you can download and unzip the datasets from below link, and you should put to the "dataset" directory.\
Then, training data are "clean_trainset_28spk_wav.zip" and "noisy_trainset_28spk_wav.zip."\
[https://datashare.ed.ac.uk/handle/10283/2791](https://datashare.ed.ac.uk/handle/10283/2791)

次に, データセットの前処理を行うため初回(もしくは再度前処理をしたい場合)はmain.pyのpreprocessing_flagをTrueに変更してください.\
numpy_dataというディレクトリが作成されその中に前処理されたデータが保存されます.\
preprocessing_flag = Falseにした場合, そのディレクトリからデータを読み込むようになります.\
When you preprocess for the first time, you need to set "preprocessing_flag" to True in main.py.
The preprocessd data is saved in "numpy_data" directory. After that, if you set the flag to False, you can load the preprocessed data from that directory.

## Model
訓練されたモデルを作成するためにはmain.py内のtraining_flagをTrueにしてください. 
trained_modelというディレクトリが作られて,その中に訓練済みのモデルが作成されます.

If you want to create a new AI model, you need to set "training_flag" to True in main.py same as preprocessing step.\
Then, "trained_model" directory is created and you can find the trained model in that directory.

また,訓練済みのモデルを使いたい場合には下記のリンクからダウンロードし,trained_modelというディレクトリ内に置いてください.\
[https://www.dropbox.com/scl/fi/iuatae6ia3t1b98xufn1c/model-30-cpu.pth?rlkey=psoii7ooqfgqq3e4ea8hfvuey&dl=0](https://www.dropbox.com/scl/fi/iuatae6ia3t1b98xufn1c/model-30-cpu.pth?rlkey=psoii7ooqfgqq3e4ea8hfvuey&dl=0)


## Visualize the spectrograms
訓練されたモデルを使い, inputディレクトリ内にあるノイズオーディオがクリーンになっている様子をスペクトログラム画像にて確認することができます.
対象となる入力データはinputディレクトリに置き, その出力スペクトログラムは outputディレクトリ内に保存されます.
入力データについて, 1secほどの短いものは使用上エラーを起こすので,なるべく5sec以上でお願いします. 

You can see how the noisy audio is denoised in the spectrogram images using trained model.
The input data is places in the "input" directory and the output spectrograms is put in the "output" directory.
The input data should be longet than 5 sec.

## References
[1] Cassia Valentini-Botinhao et al., "Noisy epeech database for training speechenhancement algorithms and tts models", University of Edinburgh. School of Informatics. Centre for Speech Technology Research (CSTR), 2017. \
[2] J. Thiemann, N. Ito, and E. Vincent, “The diverse environments multi-channel acoustic noise database: A database of multichannel environmental noise recordings,” J. Acoust. Soc. Am., vol. 133, no. 5, pp. 3591–3591, 2013.\
[3] A. E. Bulut and K. Koishida, "Low-Latency Single Channel Speech Enhancement Using U-Net Convolutional Neural Networks," ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Barcelona, Spain, 2020, pp. 6214-6218, doi: 10.1109/ICASSP40776.2020.9054563.\
[4] Stoller, D., Ewert, S., & Dixon, S. (2018). Wave-u-net: A multi-scale neural network for end-to-end audio source separation. arXiv preprint arXiv:1806.03185.\
[5] Rix, A. W., Beerends, J. G., Hollier, M. P., & Hekstra, A. P. (2001, May). Perceptual evaluation of speech quality (PESQ)-a new method for speech quality assessment of telephone networks and codecs. In 2001 IEEE international conference on acoustics, speech, and signal processing. Proceedings (Cat. No. 01CH37221) (Vol. 2, pp. 749-752). IEEE.