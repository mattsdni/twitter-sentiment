# twitter-sentiment

You will need the NLTK corpus twitter samples to run the code.
To install NLTK: run `sudo pip install -U nltk`

`import nltk`

`nltk.download()`

twitter_nb_old.py was the first file made in this project. It builds up a model based on the nltk twitter data using the naive bayes algorithm, and then prompts the user for a twitter user name, reads up to 3200 tweets using the twitter api, then graphs the data with the plot.ly api. 

twitter_nb_emoticon.py is the enhanced version of the machine learning algorithm used in the first program, with the addition of considering the emoticons heavily in the sentiment analysis. Negation of sentences is also considered.

