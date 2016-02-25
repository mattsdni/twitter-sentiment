#Matt Dennie, Tom Kreamer
#cs343 final project
#Sentiment Analysis in Python
#This program creates a sentiment analyzer based on the naive bayes algorithm
#to classify tweets as positive or negative using the training data set provided
#by the NLTK library
#A twitter user's tweets are then plotted on a bar graph over the course of
#approx. 1 year (up to 3200 tweets) using the plot.ly api

import re
import string

def remove_punctuation(s):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in s if ch not in exclude)

def tokenize(text):
    return re.split("\W+", remove_punctuation(text.lower()))

def count_words(words):
    wc = {}
    for word in words:
        wc[word] = wc.get(word, 0.0) + 1.0
    return wc

# setup some structures to store our data
vocab = {}
word_counts = {
    "pos": {},
    "neg": {}
}
priors = {
    "pos": 0.,
    "neg": 0.
}

from nltk.corpus import twitter_samples
print "Setting up text analyzer, please wait..."
tweets = twitter_samples.docs('positive_tweets.json') #Positive tweets to train model
all_tweets = []
test_tweets = []
for tweet in tweets[0:2999]:
    all_tweets.append((tweet['text'], "pos"))
for tweet in tweets[3000:4999]:
    test_tweets.append((tweet['text'], "pos"))
tweets = twitter_samples.docs('negative_tweets.json') #Negative tweets to train model
for tweet in tweets[0:2999]:
    all_tweets.append((tweet['text'], "neg"))
for tweet in tweets[3000:4999]:
    test_tweets.append((tweet['text'], "neg"))

# Build text model
for t in all_tweets:
    priors[t[1]] += 1
    words = tokenize(t[0])
    counts = count_words(words)
    for word, count in list(counts.items()):
        if word not in vocab:
            vocab[word] = 0.0
        if word not in word_counts[t[1]]:
            word_counts[t[1]][word] = 0.0
        vocab[word] += count
        word_counts[t[1]][word] += count

try:
    highpoints = re.compile(u'[\U00010000-\U0010ffff]')
except re.error:
    # UCS-2 build
    highpoints = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')
def clean_tweet(t):
    t = re.sub(r'https?:\/\/.*[\r\n]*', '', t, flags=re.MULTILINE) #remove links
    t = re.sub(r'RT','',t)      #remove RT
    t = highpoints.sub('', t)   #remove emoticons
    return t

import math
# Naive-Bayes classyfier function
def classify(tweet):
    words = tokenize(clean_tweet(tweet))   # split up into words
    counts = count_words(words) # count words, receive dict mapping words to their count

    # calculate priors, the percentage of documents in each category
    prior_neg = (priors["neg"] / sum(priors.values()))
    prior_pos = (priors["pos"] / sum(priors.values()))

    # compute probability of new doc being in each category in log space, to reduce errors
    log_prob_pos = 0.0
    log_prob_neg = 0.0

    for w, cnt in list(counts.items()):
        # skip words that we haven't seen before, or words less than 3 letters long
        if w not in vocab:
            continue

        # find probability of word in known vocab
        p_word = vocab[w] / sum(vocab.values())
        # conditional probability of this word, given the class neg
        p_w_given_neg = word_counts["neg"].get(w, 0.0) / sum(word_counts["neg"].values())
        # conditional probability of this word, given the class pos
        p_w_given_pos = word_counts["pos"].get(w, 0.0) / sum(word_counts["pos"].values())

        # compute bayesian probability
        if p_w_given_neg > 0:
            log_prob_neg += math.log(cnt * p_w_given_neg / p_word) # P(category|vocab)
        if p_w_given_pos > 0:
            log_prob_pos += math.log(cnt * p_w_given_pos / p_word) # P(category|vocab)

    results = {
        "pos": math.exp(log_prob_pos + math.log(prior_pos)),
        "neg": math.exp(log_prob_neg + math.log(prior_neg))
    }
    return results

print "Testing accuracy of model..."
test_correct = 0.
test_incorrect = 0.
for tweet in test_tweets:
    result = classify(tweet[0])
    if result["pos"] > result["neg"]:
        if tweet[1] == "pos":
            test_correct += 1
        else:
            test_incorrect += 1
    else:
        if tweet[1] == "neg":
            test_correct += 1
        else:
            test_incorrect += 1
print "Accuracy: " + str(test_correct / (test_correct + test_incorrect))
print

# get fresh tweets to classify
import twitter

consumerKey = 'F3y4afxu50QoXcy4LYZRQzzG1'
consumerSecret = 'tJhz8UyzRzl6oVzhQVLRC3kPI88zUxDmZi5EDrPaDrRLwmxTGK'
accessToken = '4311911235-TTrMARu7MwR5pdcn6Q3WUoWe3AYl0cHByQo7vAt'
accessSecret = '3rZyw2CnlcHNsGRD71yAl8YtseJc2mqAHcKvn6iLz0yeE'

#Gets the 3200 most recent tweets by user with the given username, as a list
def getTweets(username):
    api = twitter.Api(consumer_key=consumerKey,
                  consumer_secret=consumerSecret,
                  access_token_key=accessToken,
                  access_token_secret=accessSecret)
    #Get 3200 most recent tweets
    print "Loading Tweets, please wait..."
    statuses = api.GetUserTimeline(screen_name=username, count = 200, include_rts =0)
    maxId = statuses[0].id;
    for status in statuses:
        if maxId > status.id:
            maxId = status.id
    tempStatuses = []
    print "Traversing pages..."
    import sys
    for i in range(0, 15):
        tempStatuses = api.GetUserTimeline(screen_name=username, count = 200, max_id = maxId, include_rts =1)
        maxId = tempStatuses[0].id
        for tempStatus in tempStatuses:
            if tempStatus in statuses:
                pass
            if maxId > tempStatus.id:
                maxId = tempStatus.id
        statuses.extend(tempStatuses)
    print "Tweets Fetched: " + str(len(statuses))
    return statuses

positiveDates = []
negativeDates = []
def analyzeUser(username):
    print "Analyzing twitter user: " + username
    statuses = getTweets(username)

    # Calculate basic user statistics
    countPos = 0
    countNeg = 0
    for status in statuses:
        result = classify(status.text)
        if result["pos"] > result["neg"]:
            countPos += 1
            positiveDates.append(status.created_at.split()[1])
        else:
            countNeg +=1
            negativeDates.append(status.created_at.split()[1])
    print "\nUser Statistics\nPercentage of tweets that are positive: " + str(100.0* countPos / len(statuses)) + "\nPercentage that are negative: " + str(100.0*countNeg / len(statuses)) + "\n"

import plotly.plotly as py
import plotly.graph_objs as go

import numpy as np

def graphData(username):
    positiveTrace = go.Histogram(
        x=positiveDates, name = "Positive Tweet Dates"
    )
    negativeTrace = go.Histogram(
        x=negativeDates, name = "Negative Tweet Dates"
    )
    data = [positiveTrace, negativeTrace]
    layout = go.Layout(
        barmode='stack',
        xaxis=dict(
            autorange='reversed'
        )
    )
    fig = go.Figure(data=data, layout=layout)
    plot_url = py.plot(fig, filename=(username + ' Tweets By Date'),privacy='public')

cont = 'y'
while cont[0] == 'y':
    username = raw_input('Username: ')
    analyzeUser(username)
    graphData(username)
    print
    positiveDates = []
    negativeDates = []
    cont = raw_input('Analyze another user? (y/n)')[0]

import resource
print ""
print "Memory Used:"
print str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000000) + "MB"
