
# WEB-API for Pytorch NER model

This project seeks to facilitate the exchange and diffusion of NER models built with different architectures. We strongly used the code from [NCRF++](https://github.com/jiesutd/NCRFpp) which allow to build various NER neural models in pytorch.

The `tag_serve` repo adapts the ner model from [NCRF++](https://github.com/jiesutd/NCRFpp) and wrap it in a flask API to allow live demo via a web page and a deployment for medium scale production. 

## Installing from source

You can also install tag_serve from source by cloning our git repository:

`git clone https://github.com/strayMat/tag_serve.git`

Create a Python 3.6 virtual environment, and install the necessary requirements by running:

`scripts/install_requirements.sh`

Add FR_MODEL=true before the script if you want to load the spacy language modele for french.
## Call the API

### Launch API and web demo locally

+ Launch the API: `python app.py`
+ Open in the client in browser: `firefox client/pred_client.html`

### Launch API and send multiple files

+ Launch the API: `python app.py`
+ Launch call function: `python client/call.py -i decoding/ins/ -o decoding/outs/`

(add `-v` to get visualization `.html`: `python client/call.py -i decoding/ins/ -o decoding/outs/ -v`)

### With a curl command

In your terminal, run :

`curl -H 'Content-type:application/json' -d '{"file":"Paris is wonderful!"}' localhost:5000/predict`


## Change the trained model used by the API
You can either give a specific model to the api, when launching the python code `app.py` or replace the default model in the `pretrained` directory.

+ **Specify a model to `app.py`**: Launch the api with the `-m` option and specify your `new_model` name, `python app.py -m myModel/new_model` where the folder `myModel` should contain `new_model.xpt` and `new_model.model` (the architecture and the weights of the model). 
 
+ **Replace the baseline model**: Replace directly the baseline files in the pretrained directory: put new `baseline.xpt` and `baseline.model` in the `pretrained/` folder (you can check that the default model of the app is    `pretrained/baseline` by typing `python app.py --help`)


## Use docker
You can deploy the model with docker. Go on docker website to install docker and docker-compose. Then build the docker with:

`sudo docker build --build-arg http_proxy=$yourProxy -t yourTag .`

Run the docker with :

`sudo docker run -d -p 5000:5000 --name tagger yourTag python3 /app/app.py`

You can now access the docker with the previous call to the API (`client/call.py`, `client/predict_client.html` or `curl`)

## Train your model
Go see the demonstration notebook: [train_decode_template.ipynb](https://github.com/strayMat/tag_serve/blob/master/nermodel/train_decode_template.ipynb)

## References:

+ [NCRF++  : An Open-source Neural Sequence Labeling Toolkit, Yang et Zhang, 2018](https://arxiv.org/abs/1806.05626)

+ Many thanks to [Guillaume Genthial](https://guillaumegenthial.github.io/serving.html) for the excellent blog post on python web-api with flask.