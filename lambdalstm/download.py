try:
  import unzip_requirements
except ImportError:
  pass

import os
import json
import time
import tensorflow as tf
import boto3

FILE_DIR = '/tmp/'
CHECKPOINT_DIR = '/tmp/'

filelist = [ 
        f for 
        f in os.listdir(FILE_DIR) 
    ]

for f in filelist:
    os.remove(os.path.join(FILE_DIR, f))

BUCKET = os.environ['BUCKET']
LSA = 'new_lsa.pkl'
TFIDF = 'smaller_vectorizer.pkl'
ENCODER = 'encoder.pkl'
CH_DATA = 'final_sentiment.ckpt.data-00000-of-00001'
CH_INDEX = 'final_sentiment.ckpt.index'
CH_META = 'final_sentiment.ckpt.meta'

def downloadHandler(event, context):
    """
    boto3.Session(
        ).resource('s3'
        ).Bucket(BUCKET
        ).download_file(
            LSA,
            FILE_DIR+LSA)

    boto3.Session(
        ).resource('s3'
        ).Bucket(BUCKET
        ).download_file(
            TFIDF,
            FILE_DIR+TFIDF)

    boto3.Session(
        ).resource('s3'
        ).Bucket(BUCKET
        ).download_file(
            ENCODER,
            FILE_DIR+ENCODER)


    ## NOW CHECKPOINT
    boto3.Session(
        ).resource('s3'
        ).Bucket(BUCKET
        ).download_file(
            CH_DATA,
            CHECKPOINT_DIR+CH_DATA)
    boto3.Session(
        ).resource('s3'
        ).Bucket(BUCKET
        ).download_file(
            CH_INDEX,
            CHECKPOINT_DIR+CH_INDEX)
    boto3.Session(
        ).resource('s3'
        ).Bucket(BUCKET
        ).download_file(
            CH_META,
            CHECKPOINT_DIR+CH_META)
    """

    response = {
        "statusCode": 200,
        "body": json.dumps({"bucket":BUCKET})
    }
    return response
