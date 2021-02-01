# TMA
Multimodal Sentiment Analysis with Temporal Modality Attention
## First check that the requirements are satisfied:
* Python 3.6.5
* torch 1.4.0
* scikit-learn 0.23.2
* tqdm 4.54.0
* numpy 1.19.4
* visdom 0.1.8.9
## The next step is to clone the repository:
```
git clone https://github.com/qianfan1996/TMA.git
```
## put your datasets into data/CMU-MOSI, data/MMMO, data/YouTube directory
## You can run the code with:
```
python test_mosi.py 
```
in the command line. The default is to train MFN with TMA module, and gives a CMU-MOSI test set MAE of 0.960, binary classification accuracy of 77.4%.

In addition, you can change  command-line arguments: -m [mfn/tman1/marn/tman2] set model types; -t set train or test.
