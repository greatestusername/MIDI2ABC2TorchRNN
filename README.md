# MIDI2ABC2TorchRNN
Turn MIDI files into ABC notation files for use within Torch-RNN

MIDI files, in this case drum loops, are plentiful and common. Unfortunately MIDI files do not translate well when fed into an LSTM/RNN machine learning model. ABC notation is a more useful text based format for one-hot training of ML models. The process of translating standard MIDI files into ABC notation is made simple using code from the [ABC Music Project](http://abc.sourceforge.net/abcMIDI/original/).

This guide will show you how to convert your MIDI files to ABC files. Following the conversion, the ABC files can be combined into a corpus of ABC notation, which will then be used to train an LSTM model in the jcjohnson's [Torch-RNN](https://github.com/jcjohnson/torch-rnn) docker container.

Once a model has been trained, a new corpus of generated ABC notation can be split into individual ABC files, and finally converted into MIDI files. These MIDI files will be ready for playback in various media players or incorporated as elements of a musical composition in a DAW.

**Full instructions for Linux & Windows coming soon!**

## LINUX Instructions
**Installing abcmidi:**

`sudo apt-get install abcmidi`

**Using midi2abc to convert midi files to ABC notation**  
(*psssst* free midi files from: https://groovemonkee.com/pages/free-midi-loops if you need them)

`find path/to/midi/files -type f -exec  midi2abc {} -o {}.abc \;`

This uses 'find' pointed at your directory full of midis. Finding only normal file types. Executing the 'midi2abc' command on any found filename '{}' and using the '-o' option (output to file in this case) passing the filename '{}' and extension '.abc'. Then stop searching '\;' and process is done.

Example:  
`find /home/brains/Downloads/midi -type f -exec  midi2abc {} -o {}.abc \;`

**Next: navigate to your directory with the new .abc files in it. Combine the contents of all abc files into one corpus**

`cat *.abc > midi-corpus.txt`

Here 'cat' is being called on all '*' .abc files and routing '>' the output to 'midi-corpus.txt'
You can view the file normally and you should see many patterns one after another in the ABC notation format

ABC Notation format example:
> X: 1  
> T: from ./095 New Day A.mid  
> M: 4/4  
> L: 1/8  
> Q:1/4=95  
> % Last note suggests Lydian mode tune  
> K:G % 1 sharps  
> % 095 New Day A  
> V:1  
> %%MIDI channel 10  
> % 095 New Day A  
> % 095 New Day A  
> xx xx xx/2x/2 xx/2x/2| \  
> xx xx xx/2x/2 xx/2x/2| \  
> xx xx xx/2x/2 xx/2x/2| \  
> xx xx/2x/2 x/2x/2x/2x/2 xx/2  

**Install Docker**  
Check for Docker install instructions for your distribution here: https://docs.docker.com/engine/installation/linux/

Or if you're lazy and trust Docker to do the work for you:

`curl -sSL https://get.docker.com/ | sh`

**Run the Torch-rnn docker image created by Cristian Baldi (https://github.com/crisbal/docker-torch-rnn)**

`docker run -ti crisbal/torch-rnn:base bash`

This will start 'docker' download and 'run' the docker container 'crisbal/torch-rnn:base' with the 'bash' command line
using the options '-ti' which will put you in the docker container's interactive console

_If the image is running already from last time you can find the container ID_  
`docker ps`  
_Copy the CONTAINER ID (usually a string of numbers/letters) and attach with_  
`docker attach YOURCONTAINERID`  

**Next: Copy the midi corpus file to the docker container for processing.**  
(this command should be entered from your normal console, not the docker container torch-rnn console)

`docker cp /path/to/your/midi-corpus.txt CONTAINERID:/root/torch-rnn/data/midi-corpus.txt`

This tells docker to copy 'cp' the midi-corpus.txt file to the chosen Container ID in the /root/torch-rnn/data/ directory

Example:  
`docker cp home/brains/Downloads/midi/midi-corpus.txt c849b04dd3f8:root/torch-rnn/data/midi-corpus.txt`

**Preprocess the midi-corpus in the docker container file using included script:**  

`python scripts/preprocess.py --input_txt data/midi-corpus.txt --output_h5 data/midi-corpus.h5 --output_json data/midi-corpus.json`

**Train a model using your new preprocessed data from the corpus**  
(This will train using only the CPU which may be slow. To use GPU install CUDA and use nvidia-docker instead of docker)

`th train.lua -input_h5 data/midi-corpus.h5 -input_json data/midi-corpus.json -seq_length 900 -max_epochs 300 -checkpoint_name cv/pickacheckpointname`

This will call the lua script 'train.lua' with the the h5 and json inputs you created in the last step.  
The sequence_length of 900 ensures 900 characters are iterated over. This makes sure ABC notation patterns under 900 characters are fully trained (feel free to try other sizes).  
The max_epoch 300 is an arbitrary number to make sure enough training generations are done (feel free to try other sizes)  
The final option sets the checkpoint name prefix that every checkpoint will be named from in the cv/ folder  
For more training options and full descriptions see: https://github.com/jcjohnson/torch-rnn/blob/master/doc/flags.md#training

Epochs should scroll by with a format similar to:  
> Epoch 1.01 / 300, i = 1 / 20100, loss = 1.277552  
> Epoch 1.03 / 300, i = 2 / 20100, loss = 1.484522   
> Epoch 1.04 / 300, i = 3 / 20100, loss = 1.338697  
> ...  

Your 'loss' may be different depending on the size of your corpus and training options.  
Everytime 'i' equals a number divisible by 1000 you should see a val_loss printed.

The val_loss looks like this:  
> Epoch 15.93 / 300, i = 1000 / 20100, loss = 0.217981  
> val_loss = 	0.43221498653293  

**After all epochs of training are done note which iteration has the lowest val_loss.**  
Generate some sampled output using the checkpoint file with the lowest val_loss  
(In this example case 11000 has the lowest val_loss)

`th sample.lua -checkpoint cv/yourchosencheckpointname_11000.t7 -sample 1 -temperature 0.71 -start_text "X: 1" -length 9900 > midioutput.txt`

This will run the 'sample.lua' script over the checkpoint file in the 'cv/' directory named 'yourchosencheckpointname_11000.t7'  
Telling the script to 'sample' (generate) some data with a 'temperature' of '0.71'  
The 'start_text' of 'X: 1' (the start of an ABC notation pattern) will seed the data generation producing a file of ~9900 characters that is routed '>' to a file named 'midioutput.txt'

The file midioutput.txt contains the newly generated abc notation patterns.  
If you view the file you should see NEW ABC patterns similar to the ABC notation example above.

**Copy the midioutput.txt file to your host desktop**

`docker cp CONTAINERID:root/torch-rnn/midioutput.txt /host/path/to/copy/file/to/midioutput.txt`

Example:  
`docker cp c849b04dd3f8:root/torch-rnn/midioutput.txt /home/brains/Downloads/lstm-midis/midioutput.txt`

**Next: split the midioutput.txt**  
Split the file at the start of each ABC pattern (denoted by "X:") to produce individual ABC pattern files with a .abc extension

`awk '/X:/{x="F"++i".abc";}{print > x;}' midioutput.txt`

This will create a number of files starting with F and ending with an .abc extension in the same directory as your midioutput.txt file.

**Now convert the individual .abc files to midi files**

`find /host/path/to/your/new/abcfiles/ -type f -exec abc2midi {} -o {}.mid \;`

Example:  
`find /home/brains/Downloads/lstm-midis/ -type f -exec abc2midi {} -o {}.mid \;`

You should now have a number of newly generated MIDI files roughly equal to the number of .abc files in the directory.

Enjoy your new midis! Experiment with midis other than drums! You can even run poems/books/etc through the torch-rnn process! Generate all the things (as long as they are text based)!
