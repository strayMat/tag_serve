
# WEB-API for Pytorch NER model

This project seeks to facilitate the exchange and diffusion of NER models built with different architectures. We strongly used the code from [NCRF++](https://github.com/jiesutd/NCRFpp) which allow to build various NER neural models in pytorch.

The `torch_serve` repo adapts the ner model from [NCRF++](https://github.com/jiesutd/NCRFpp) and wrap it in a flask API to allow live demo via a web page and a deployment for medium scale production (upcoming). 

## Launch API and web demo locally

+ Launch the API: `python app.py`
+ Open in the client in browser: `firefox pred_client.html`


## References:

+ [NCRF++  : An Open-source Neural Sequence Labeling Toolkit, Yang et Zhang, 2018](https://arxiv.org/abs/1806.05626)

+ Many thanks to [Guillaume Genthial](https://guillaumegenthial.github.io/serving.html) for the excellent blog post on python web-api with flask.