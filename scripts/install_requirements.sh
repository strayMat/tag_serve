#!/bin/bash

pip install -r requirements.txt
python -m spacy download en

# only install fr model if specifically asked
if [[ "$FR_MODEL" == "true" ]]; then
    python -m spacy download fr
fi