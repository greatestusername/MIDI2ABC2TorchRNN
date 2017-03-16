# MIDI2ABC2TorchRNN
1. Turn MIDI files into ABC notation corpus
2. Use ABC notation corpus to train a model in Torch-RNN
3. Generate new ABC corpus from trained model
4. Split generated ABC corpus into MIDI files
5. ???
6. Profit

MIDI files, in this case drum loops, are plentiful and common. Unfortunately MIDI files do not translate well when fed into an LSTM/RNN machine learning model. ABC notation is a more useful text based format for one-hot training of ML models. The process of translating standard MIDI files into ABC notation is made simple using code from the [ABC Music Project](http://abc.sourceforge.net/abcMIDI/original/).

This guide will show you how to convert your MIDI files to ABC files. Following the conversion, the ABC files can be combined into a corpus of ABC notation, which will then be used to train an LSTM model in the [Torch-RNN](https://github.com/jcjohnson/torch-rnn) docker container put together by jcjohnson.

Once a model has been trained, a new corpus of generated ABC notation can be split into individual ABC files, and finally converted into MIDI files. These MIDI files will be ready for playback in various media players or usable as elements in a musical composition.

**Full instructions for Linux & Windows coming soon!**

## LINUX Instructions
**Installing abcmidi:**

`sudo apt-get install abcmidi`

**Using midi2abc to convert midi files to ABC notation**  
(*psssst* free midi files from: https://groovemonkee.com/pages/free-midi-loops if you need them)

`find path/to/midi/files -type f -exec  midi2abc {} -o {}.abc \;`

This uses 'find' pointed at your directory full of midis. Finding only normal file types. Executing the 'midi2abc' command on any found filename '{}' and using the '-o' option (output to file in this case) passing the filename '{}' and extension '.abc'. Then stop searching '\;' and process is done.

Example:  
`find /home/greatestusername/Downloads/midi -type f -exec  midi2abc {} -o {}.abc \;`

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

This will start 'docker' and download, as well as, 'run' the docker container 'crisbal/torch-rnn:base' with the 'bash' command line using the options '-ti' which will open the docker container's interactive console

**Next: Copy the midi corpus file to the docker container for processing.**  
(this command should be entered from your normal console, not the docker container torch-rnn console)

`docker cp /path/to/your/midi-corpus.txt CONTAINERID:/root/torch-rnn/data/midi-corpus.txt`

This tells docker to copy 'cp' the midi-corpus.txt file to the chosen Container ID in the /root/torch-rnn/data/ directory

Example:  
`docker cp home/greatestusername/Downloads/midi/midi-corpus.txt c849b04dd3f8:root/torch-rnn/data/midi-corpus.txt`

**Preprocess the midi-corpus in the docker container file using included script:**  

`python scripts/preprocess.py --input_txt data/midi-corpus.txt --output_h5 data/midi-corpus.h5 --output_json data/midi-corpus.json`

**Train a model using your new preprocessed corpus data**  
(This will train using only the CPU which may be slow. For GPU processing install CUDA and use nvidia-docker instead of docker)

`th train.lua -input_h5 data/midi-corpus.h5 -input_json data/midi-corpus.json -seq_length 900 -max_epochs 300 -checkpoint_name cv/pickacheckpointname`

Start training with 'train.lua' using the the h5 and json inputs you created in the last step.  
The sequence_length of 900 ensures 900 characters are iterated over. This makes sure longer ABC notation pattern parameters are trained into the model (feel free to try other sizes).  
The max_epoch 300 is an arbitrary number to make sure enough training generations are done (feel free to try other numbers)  
The final option sets the checkpoint name prefix that every checkpoint will be named from in the cv/ folder  
For more training options and full descriptions which may improve your training results see: https://github.com/jcjohnson/torch-rnn/blob/master/doc/flags.md#training

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

After all epochs of training are done use the iteration with the lowest val_loss.

**Generate some sampled output using the checkpoint file with the lowest val_loss**  
(In this example case 11000 has the lowest val_loss)

`th sample.lua -checkpoint cv/yourchosencheckpointname_11000.t7 -sample 1 -temperature 0.71 -start_text "X: 1" -length 9900 > midioutput.txt`

This will run the 'sample.lua' script over the checkpoint file in the 'cv/' directory named 'yourchosencheckpointname_11000.t7'  
Telling the script to 'sample' (generate) some data with a 'temperature' (the amount of randomness between 0 and 1) of '0.71'  
The 'start_text' of 'X: 1' (the start of an ABC notation pattern) will seed the data generation.  
The sample output will be stopped at '9900' characters then routed '>' to a file named 'midioutput.txt'

The file midioutput.txt contains the newly generated abc notation patterns.  
If you view the file you should see NEW ABC patterns similar to the ABC notation example pattern shown earlier.

**Copy the midioutput.txt file from the docker container to a local folder**

`docker cp CONTAINERID:root/torch-rnn/midioutput.txt /host/path/to/copy/file/to/midioutput.txt`

Example:  
`docker cp c849b04dd3f8:root/torch-rnn/midioutput.txt /home/greatestusername/Downloads/lstm-midis/midioutput.txt`

**Next: split the midioutput.txt**  
Split the file at the start of each ABC pattern (denoted by "X:") to produce individual ABC pattern files with an .abc extension

`awk '/X:/{z="M"++i".abc";}{print > z;}' midioutput.txt`

Here 'awk' is searching 'midioutput.txt' for a pattern starting with 'X:' (the start of an .abc file header)  
Each pattern starting with 'X:' creates an entry with the name M#.abc  
The pattern awk has found is then routed '>' to 'z' which is equal to the newly named M#.abc file

The output of this command will create a number of files starting with "M", followed by the file number, and ending with an .abc extension in the same directory as your midioutput.txt file.

**Now convert the individual .abc files to midi files**

`find /host/path/to/your/new/abcfiles/ -type f -exec abc2midi {} -o {}.mid \;`

Example:  
`find /home/greatestusername/Downloads/lstm-midis/ -type f -exec abc2midi {} -o {}.mid \;`

This is the midi2abc step in reverse.

You should now have a number of newly generated MIDI files roughly equal to the number of .abc files in the directory.

Enjoy your new MIDIs! Experiment with MIDIs other than drums! You can even run poems/books/etc through Torch-RNN to train other models! Generate all the things (as long as they are text based)!
