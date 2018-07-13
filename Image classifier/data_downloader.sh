#!/usr/bin/env bash

set -e
# Download Kaggle Cats and Dogs Dataset.
wget https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip
unzip kagglecatsanddogs_3367a.zip -d kagglecatsanddogs
chmod +x kagglecatsanddogs

