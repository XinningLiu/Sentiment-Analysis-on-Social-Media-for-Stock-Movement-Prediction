# Sentiment-Analysis-on-Social-Media-for-Stock-Movement-Prediction
Analyzed sentimental information of Yahoo Finance Conversation messages to forecast stock movements using 4 sentiment analysis algorithms in Python

webcrawler.py: Download the messages from Yahoo Finace Conversation board of stock with sticker "T". You will need to install a chrome driver on your chomputer and change the address of the driver in the code which is in line 20. You also can change the sticker to other stocks so that you can download other stocks' messages.

message.py: Contains the basic stucture of the sentiment analysis algorithms. No need to change any line.

sentimentAnalysis.py: This file contains following functions: 1. read the xml of the Stanford CoreNLP output; 2. 4 sentiment analysis algorithms, i.e. LDA based, JST based, Aspected based ( an algorithm proposed by the paper Sentiment Analysis on Social Media for Stock Movement Prediction) and NLP based; 3. compute the sentiment score for forcasting stock movements; 4. write and read function of LDA and JST theta matrix, which is used for interprocesser communication.

Other files: scripts for Stanford CoreNLP.
