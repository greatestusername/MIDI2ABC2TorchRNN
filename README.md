# MIDI2ABC2TorchRNN
Turn MIDI files into ABC notation files for use within Torch-RNN

MIDI files, in this case drum loops, are plentiful and common. Unfortunately MIDI files do not translate well when fed into an LSTM/RNN machine learning model. ABC notation is a more useful text based format for one-hot training of ML models. The process of translating standard MIDI files into ABC notation is made simple using code from the [ABC Music Project](http://abc.sourceforge.net/abcMIDI/original/).

This guide will show you how to convert your MIDI files to ABC files. Following the conversion, the ABC files can be combined into a corpus of ABC notation, which will then be used to train an LSTM model in the jcjohnson's [Torch-RNN](https://github.com/jcjohnson/torch-rnn) docker container.

Once a model has been trained, a new corpus of generated ABC notation can be split into individual ABC files, and finally converted into MIDI files. These MIDI files will be ready for playback in various media players or incorporated as elements of a musical composition in a DAW.

**Full instructions for Linux & Windows coming soon!**
