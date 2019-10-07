# Mortgage Document Classification Assignment
## Overview
I built a webservice for classifying mortgage documents. The system was trained on the data given for this assignment, incorporating a few approaches and machine learning techniques. When a respectable accuracy was obtained I saved the binaries of these models and incorporated the prediction process into an API running on Django. I also built a frontend in Angular.js so that this API could be accessed easily by anyone with a browser.

## But first...
I wanted to mention that this assignment was a pleasure to work, and I would love the opportunity to explain my solution in person.

## Live Demo
**The document classifying webservice lives at: http://34.220.107.85/**

**Or, if you prefer, a much more accurate version at: http://54.213.175.162/**
![Actual Site](http://boazreisman.com/images/classifier_demo.png)

## Explanation of the Machine Learning
A detailed description of my thought process in developing this service can be found in the files included at the root level in this github repository.
- classification_exercise.html
- classifcation exercise.pdf

- lstm_accuracy.html
- lstm_accuracy.pdf

I will attempt to summarize that process and my conclusions here.
- The data contained about 62K rows with the categories not at all evenly distributed, this led me to believe that a probabilistic model might produce good results.
- Upon looking at the total word count I found that while there were ~ 2 million words used in all the documents combined, there were ~ 1 million unique words, clearly there were a lot of specifics to each document, and I would have to remove these (as well as the very common stop words).
- I first tried a Naive Bayes, even with eliminating stop words and infrequent words, and allowing for an N-gram range of 1-3 (using single words and up to 3 word phrases), the accuracy was not high enough to move on.
- I used the technique of Latent Semantic Analysis. First I factorized a matrix of TF-IDF vectors and truncated this factorization to a 100 dimension space. Projecting my vectors into this reduced space, I was able to cluster them using the K-Nearest Neighbor algorithm, and finally, classify my test set using proximity to these clusters. Using this technique I got an accuracy of 82.6%
- Clearly the order of the words in the original documents is something important, and if I could continue working on this assignment, I would use an LSTM deep learning model to try to improve this accuracy.

# Webservice and API
The webservice itself is built in Django, the root path returns an single page web app built in angular.js.

The page is just a textarea and a submit button. If you paste some of the anonymized "words" into the textarea, the angular hist the api endpoint with a GET, and the words are htmlencoded into the request (per the assignments instructions).

The Api has a service which uses the binaries of the trained models and. This service is a singleton pattern which only needs to load once on app startup, it then can call the predictor function quickly from the api endpoint.

The returned JSON is displayed on the page: prediction category and confidence.

# Deployment
The webservice was deployed by placing the Django application into a Docker and deploying it to an AWS EC2 instance. This was the most straightforward solution to demoing a simple service.

In the future, a more robust deployment would include the following AWS services.
- The exported models would live on S3.
- When placed there or updated, the AWS Lambda, version of the predictor code would update and incorporate. This could also be used to retrain, when more data is added.
- Instead of living on an Ubuntu service, easiest and most scalable solution would be to place the container in an Elastic Beanstalk or ECS instance.
- Finally Codepipeline could be used as a continuous delivery service. So that code updates, if they passed tests, would not need to deployed by hand.

# Parting thoughts
As I mentioned, this assignment was a blast to work on. If at all possible I would love to talk to the team and explain my solutions further, and my next steps would be to use a LSTM model to further improve the accuracy of the classifier and to use other AWS service to build a more robust infrastructure for the tool.

