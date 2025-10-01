#!/bin/bash

echo "REQUIRE <curl> <unzip> and <mv>"
echo "DOWNLOADING THE DATASET FROM KAGGLE USING <curl> ......"
echo "ALWAYS CLICK [y]es 🥹"

curl -L -o ../data/data.zip  https://www.kaggle.com/api/v1/datasets/download/mlg-ulb/creditcardfraud

unzip ../data/data.zip -d ../data/

mv ../data/creditcard.csv ../data/data.csv