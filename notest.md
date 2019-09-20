# Don't forget this in the email
Put this in there: You reading my commit messages?
Good that shows the right kind of digging.
Make sure to mention this in your email.
Cheers.

# Data Science
Histogram of labels
[('BILL', 18968),
 ('POLICY CHANGE', 10627),
 ('CANCELLATION NOTICE', 9731),
 ('BINDER', 8973),
 ('DELETION OF INTEREST', 4826),
 ('REINSTATEMENT NOTICE', 4368),
 ('DECLARATION', 968),
 ('CHANGE ENDORSEMENT', 889),
 ('RETURNED CHECK', 749),
 ('EXPIRATION NOTICE', 734),
 ('NON-RENEWAL NOTICE', 624),
 ('BILL BINDER', 289),
 ('INTENT TO CANCEL NOTICE', 229),
 ('APPLICATION', 229)]


Describe the word count per document:
count    62204.000000
mean       334.148479
std        330.217525
min          0.000000
25%        148.000000
50%        252.000000
75%        402.000000
max       9076.000000
Name: word_count, dtype: float64

Total words (not deduped): 20785372
Total unique words: 1037934

Vocab size (not incredibly rare words): 87998
Words in vocab, describe:
count    62204.000000
mean       101.385120
std         86.443821
min          0.000000
25%         48.000000
50%         77.000000
75%        128.000000
max       1610.000000
Name: words_in_vocab, dtype: float64

More managable distribution of words
Train naive bayes with an n_gram range of 1 - 3


naive_bayes training vector:
(49727, 10000)

Those are still incredibly high dimensions

Using SVD to reduce dimensions, and then projecting into that space
K-Nearest Neighbors classifiers: 82% accurate

Distribution of words


Remove Stop words
Remove Words that are too frequent

# Naive Bayes
With TF-IDF
Cross Validation (K-Folds)
Results

# Linear Model

# Word Vectors
Word2Vec
SVD

# Sequence to Label

# Deployment
ML - AWS Lambda
API - Docker on EC2
Frontend - Angular.js

# Documentation
README.md

# Screencast

