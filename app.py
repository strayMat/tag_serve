import os
import sys
import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import spacy
import json 

from spacy import displacy
from serve import get_model_api
from nermodel.utils.tokenizer import myTokenizer

VISU_SERVE = False
VISU_SAVE = True
VISU = VISU_SAVE or VISU_SERVE
LANGUAGE = 'en'
path2xpt = 'pretrained/baseline.xpt'
path2model = 'pretrained/baseline.model'

# define the app
app = Flask(__name__)
CORS(app) # cross-domain requests, allow everything by default 

# loading model once and for all the api
model_api = get_model_api(path2xpt, path2model)

# loading tokenizer once and for all for the api
app.logger.info('Loading spacy tokenizer...')
tokenizer = myTokenizer(LANGUAGE)

#STATUS = 'live' # live/file

# API live demo route
@app.route('/api', methods=['POST'])
def api():
    input_data = request.json
    app.logger.info('api_input: ' + str(input_data))
 	
    input_client, output_client = model_api(input_data, tokenizer = tokenizer)
    app.logger.info('api_output: ' + str(output_client))
    response = jsonify(input= input_client, output = output_client)
    return response

# API live predict 
## commandline
# curl -X POST -F file=@prod_data/wiki_en_france.txt 'http://localhost:5000/predict'
@app.route('/predict', methods =['POST'])
def file_api():
    # upload a file
    #file = '../prod_data/wiki_en_france.txt'
    if request.method == 'POST':
        VISU_SERVE = False
        VISU_SAVE = True
        VISU = VISU_SAVE or VISU_SERVE
        # read configurations for the predict function
        
        if request.files.get('conf'):
            conf = json.loads(request.files["conf"].read())
            VISU_SAVE = conf['visu']
            VISU = VISU_SAVE or VISU_SERVE
        
        # read the file path
        if request.files.get("file"):
            # read the image in PIL format
            input_data = request.files["file"].read().decode('utf-8')
            filename = request.files["filename"].read().decode('utf-8')
            app.logger.info('api input file:' + str(filename))
            #app.logger.info('api_input: ' + str(input_data))
            
            # Predict
            input_client, output_client = model_api(input_data, live = False, tokenizer = tokenizer)
            # app.logger.info('api_output: ' + str(output_client))
            # post processing of the data:
            out = [' '.join(sent) + '\n' for sent in output_client]

            if VISU:
                entities = []
            annotations = []
            idx = 1
            text = tokenizer.tokenize(input_data)
            for sent, label_seq in zip(text, output_client):
                for token, label in zip(sent, label_seq):
                    if label != "O":
                        new_ann = 'T'+str(idx)+'\t'+label[2:]+' '+str(token.idx)+' '+str(token.idx+len(token.string.strip()))+'\t'+token.string.strip()+'\n'
                        annotations.append(new_ann)
                        idx+=1
                        if VISU:
                            entities.append({'start':token.idx, 'end':token.idx+len(token.string.strip()), 'label':label[2:]})
            
            html = None
            if VISU:
                visu_data = [{'text':input_data, 'ents':entities, 'title':None}]
                if VISU_SAVE:
                    html = displacy.render(visu_data, style='ent', page=True, manual=True)
                if VISU_SERVE:
                    displacy.serve(visu_data, style='ent', manual=True, port=5001)
            response = jsonify(input_data=input_data, annotations=annotations, html=html)

            return response


# default route
@app.route('/')
def index():
    return "Index API: tag_serve is on fire!"

# Http errors handlers 
@app.errorhandler(404)
def url_error(e):
    return """
    WRONG URL!
    <pre>{}<\pre>""".format(e), 404

@app.errorhandler(500)
def server_error(e):
    return """
    An internal error occured: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    # used when running locally
    app.run(host='0.0.0.0', debug = True)
