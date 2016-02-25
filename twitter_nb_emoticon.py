"""



                                                                     `.........
                                                                    `dNNNNNNNNm:
                                                                    sMMMMMMMMMMm`
                                                                   /MMMMMMMMMMMMy
                                                                  .NMMMMMMMMMMMMM+
                                                                 `dMMMMMMsyMMMMMMN-
                                                                 sMMMMMMd``dMMMMMMm`
                                                                /MMMMMMN-  -NMMMMMMy
                                                               .NMMMMMM+    +MMMMMMM+
                                                              `dMMMMMMy      hMMMMMMN-
                                                              sMMMMMMm`      .mMMMMMMm`
                                                             /MMMMMMN:        :MMMMMMMy
                                                            .NMMMMMMo          sMMMMMMM+
                                                           `hMMMMMMh`          `dMMMMMMN-
                                                           oMMMMMMN.            -NMMMMMMd`
                                                          :MMMMMMM/              +MMMMMMMs
                                                         .mMMMMMMy                yMMMMMMM/
                                                         hMMMMMMm`                `mMMMMMMN.
                                                        oMMMMMMM-                  :MMMMMMMd`
                                                       -NMMMMMMo                    oMMMMMMMs
                                                      `mMMMMMMh                     `dMMMMMMM:
                                                      yMMMMMMN.                      .NMMMMMMN.
                                                     +MMMMMMM/                        /MMMMMMMh
                                                    -NMMMMMMy                          yMMMMMMMo
                                                   `mMMMMMMm`                          `mMMMMMMM:
                                                   yMMMMMMN-                            -MMMMMMMm`
                                                  /MMMMMMM+                              oMMMMMMMh
                                                 .NMMMMMMh                                hMMMMMMM+
                                                `dMMMMMMN.                                .NMMMMMMM-
                                                /hhhhhhh:                                  :hhhhhhhs



                              s`                  `s+       /s-        -s/ .yoooooooooo/  `hooooooo+/.         .y:
                              d.                 `y+y+      oyoo-    -ooyo -h         `so .m`     ``:so`      .h/h:
                              d.                `y+ `y+     oo -oo-.oo- oo -h         `so .m          oo     .h: .h-
                              d.               `y+   `y+    oo   -oo-   oo -mooooooooohy` .m          -d    .h:   .h:
                              d.              `yo`````.h+   oo          oo -h         .s+ .m          os   .h/`````-d-
                              d.             `sy+++++++oh+  oo          oo -h         `so .m       `-oo`  .hs+++++++sh-
                              yooooooooooo+` /+         `s. //          // -hoooooooooo/` `hooooooo+/.    o:         .y`


                              -::/::- /::::::  .-:::: :     - :.    -` .-::.` :        .-::-`  .-::::`: /:::::- .::::::
                                 s    s`````` /:`     y`````s`s:/.  +`/:` `-o s       :/` `-+`:/````` s s`````` s.````
                                 s    y------ s`      y-----y`s `:/.+`o`    o`s       o`    +.o. `--+-s y------ .-----+
                                 o    s...... ./:.... o     +`o   `:s`./:--/: o-..... `/:--/: `/:-..+-o s...... .....-o
                                 `    ```````   ````` `     ` `     `   ```   ```````    ``      ```` ` ``````` ``````
"""
#Matt Dennie
#Sentiment Analysis in Python
#
# Features Implemented
# [x] Negation handling: When a sentence is negated (don't, can't, wouldn't etc.) this is taken into account
# [x] Handle most horzontal emoticons eg:  >:D  :((
# [ ] Consider word case eg: all caps
# [ ] Amplify emoticon sentiment eg: :( is not as sad as :(((
# [ ] Record n-grams for better context understanding (n-gram match score is s^n)
# [ ] Feature selection/triming via mutual information function (to maximize valuable features, minimize trivial ones)
# [ ] Face/Emoticon detection and classification using eigenfaces, then k means clustering, then machine learning with Naive-Bayes

import re
import string

def remove_punctuation(s):
	exclude = set("'.!")
	return ''.join(ch for ch in s if ch not in exclude)

def tokenize(text):
	return re.split("\s+", remove_punctuation(text.lower()))

def count_words(words):
	wc = {}
	for word in words:
		wc[word] = wc.get(word, 0.0) + 1.0
	return wc

#Detects negations and transforms negated words into "not_" form.
def negate(text):
	negation = False
	delims = "?.,!"
	result = ""
	words = text.split()
	for word in words:
		stripped = word.strip(delims).lower()
		negated = "not_" + stripped if negation else stripped
		result += negated + " "

		if any(neg in word for neg in ["not", "n't"]):
			negation = not negation

		if any(c in word for c in delims):
			negation = False

	return result

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
for tweet in tweets[2000:4999]:
	all_tweets.append((tweet['text'], "pos"))
for tweet in tweets[0:1999]:
	test_tweets.append((tweet['text'], "pos"))
tweets = twitter_samples.docs('negative_tweets.json') #Negative tweets to train model
for tweet in tweets[2000:4999]:
	all_tweets.append((tweet['text'], "neg"))
for tweet in tweets[0:1999]:
	test_tweets.append((tweet['text'], "neg"))

# Build text model
for t in all_tweets:
	priors[t[1]] += 1
	words = tokenize(negate(t[0]))
	counts = count_words(words)
	for word, count in list(counts.items()):
		if word not in vocab:
			vocab[word] = 0.0
		if word not in word_counts[t[1]]:
			word_counts[t[1]][word] = 0.0
		vocab[word] += count
		word_counts[t[1]][word] += count

def find_emoticons(s):
	reg = '(\:\w+\:|\<[\/\\]?3|[\(\)\\\D|\*\$]+[\-\^]?[\:\;\=]|[\:\;\=B8][\-\^]?[3DOPp\@\$\*\\\)\(\/\|]+)(?=\s|[\!\.\?]|$)'
	return re.findall(reg, s)

def clean_tweet(t):
	t = re.sub(r'https?:\/\/.*[\r\n]*', '', t, flags=re.MULTILINE) #remove links
	t = re.sub(r'RT','',t)	  #remove RT
	emoti = find_emoticons(t)
	for e in emoti:
		for i in range(0, 10):
			t+= ' '+e
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

def test():
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
				#print tweet
				#print
		else:
			if tweet[1] == "neg":
				test_correct += 1
			else:
				test_incorrect += 1
				#print tweet
				#print
	print "Accuracy: " + str(test_correct / (test_correct + test_incorrect))
	print

test()


import resource
print ""
print "Memory Used:"
print str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000000) + "MB"
