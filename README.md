
# WEB-API for Pytorch NER model


This project seeks to facilitate the exchange and diffusion of NER models built with different architectures. We strongly used the code from [NCRF++](https://github.com/jiesutd/NCRFpp) which allow to build various NER neural models in pytorch.

The `tag_serve` repo adapts the ner model from [NCRF++](https://github.com/jiesutd/NCRFpp) and wrap it in a flask API to allow live demo via a web page and a deployment for medium scale production. 

## Installing from source

You can also install tag_serve from source by cloning our git repository:

`git clone https://github.com/strayMat/tag_serve.git`

Create a Python 3.6 virtual environment, and install the necessary requirements by running:

`scripts/install_requirements.sh`

Add FR_MODEL=true before the script if you want to load the spacy language modele for french.

## Launch API and web demo locally

+ Launch the API: `python app.py`
+ Open in the client in browser: `firefox client/pred_client.html`

## Launch API and send multiple files

## Train your model

## References:

+ [NCRF++  : An Open-source Neural Sequence Labeling Toolkit, Yang et Zhang, 2018](https://arxiv.org/abs/1806.05626)

+ Many thanks to [Guillaume Genthial](https://guillaumegenthial.github.io/serving.html) for the excellent blog post on python web-api with flask.