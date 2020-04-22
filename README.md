Keyword spotter train

## Installation
sudo apt install libportaudio2
python3.8 -m pip install sounddevice librosa torchaudio

## Notes
_background_noise_ is shared between train/dev/test. This possibly leads to overfitting, but I'm reproducing [honk](https://github.com/castorini/honk/blob/master/utils/model.py#L339) code.
