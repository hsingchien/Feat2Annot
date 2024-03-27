LSTM encoder-decoder model to "translate" posture tracking sequence to behavior annotations.  
#### Basic design
Adapted from the LSTM encoder-decoder seq2seq model. Starts with a bidirectional LSTM encoder which encode the source sequence. The last output of encoder is then fed as the initial hidden and cell into the decoder LSTM. Meanwhile, the hidden state of encoder and decoder are used to compute a global attention score. Attention output and decoder hidden is combined and fed into linear layer to compute the probability scores of annotation. 

#### Current problem
The training dataset is heavily unbalanced. Vast majority of annotations are 'other'. The model output is flooded by 'other', which achieves decent accuracy.  
I plan to address this problem by  
* Balance the dataset.
* Modify the neural network. Train a separate FC network which takes features and output annotation. Use final layer output as input to the seq2seq model. 
* Adjust loss function to put more weights on. 