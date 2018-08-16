import os
import sys
import logging
import optparse

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


# parser
parser = optparse.OptionParser()
parser.add_option('-m', '--model', default=os.path.join('pretrained', 'baseline'), help='path/name of the pretrained model to use (default: pretrained/baseline), the conll2003 pretrained model')
parser.add_option('-l', '--language', default='fr', help='Spacy language model for tokenizer (default: en)')
option, args = parser.parse_args()
modelpath = option.model
LANGUAGE = option.language

path2xpt = modelpath+'.xpt'
path2model = modelpath+'.model'

# define the app
app = Flask(__name__)
CORS(app) # cross-domain requests, allow everything by default 

# loading model once and for all the api
model_api = get_model_api(path2xpt, path2model)

# loading tokenizer once and for all for the api
print('Loading spacy tokenizer in '+ LANGUAGE+'....')
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
    if request.method == 'POST':
        form = 'min'
        VISU = False
        # Loading data
        data = request.json
        print(data)
        # If requested by python.requests.post (from call.py)
        if data == None:
            data = request.files
            input_data = data["file"].read().decode('utf-8')
            conf = json.loads(data["conf"].read().decode('utf-8'))
        else:
            input_data = data["file"]
        # read configurations for the predict function
        if "conf" in data:
            if conf['visu'].lower() == "true":
                VISU == True
            if conf['format'].lower() == 'brat':
                form = 'brat'
        # read file for the predict function      
        if "file" in data:
            app.logger.info('**********Decoding file**********')
            app.logger.info('Save html visu: ' + str(VISU))
            app.logger.info('Annotation formats: ' + str(form))
            app.logger.info('Api input (50 first caracters): ' + str(input_data[:50]))
            # Predict
            input_client, output_client = model_api(input_data, live = False, tokenizer = tokenizer)
            # app.logger.info('api_output: ' + str(output_client))
            # post processing of the data:
            out = [' '.join(sent) + '\n' for sent in output_client]
            text = tokenizer.tokenize(input_data)
            # build annotations outputs
            annotations, entities = build_ann(text, output_client, visu = VISU, form = form)
            
            html = None
            if VISU:
                visu_data = [{'text':input_data, 'ents':entities, 'title':None}]
                html = displacy.render(visu_data, style='ent', page=True, manual=True)
            
            app.logger.info('First predicted entity: ' + str(annotations[1]))
            response = jsonify(input_data=input_data, annotations=annotations, html=html)    
            return response


        
def build_ann(sent_list, ann_list, visu = False, form = 'min'):
    entities = []
    annotations = []
    idx = 0
    start = -1
    end = -1
    entity = None
    string = None            
    for sent, label_seq in zip(sent_list, ann_list):
        for token, label in zip(sent, label_seq):
            if label != "O":
                if label[0] == 'B':
                    # add previous entity
                    if entity is not None:
                        if form == 'brat':
                            new_ann = 'T'+str(idx)+'\t'+entity+' '+str(start)+' '+str(end)+'\t'+string+'\n'
                        elif form == 'min':
                            new_ann = entity+';'+str(start)+';'+str(end)+'\n'
                        annotations.append(new_ann)
                        idx+=1
                        if visu:
                            entities.append({'start':start, 'end':end, 'label':entity})
                    # re-initalize entity
                    start = token.idx
                    end = token.idx + len(token.string.strip())
                    entity = label[2:]
                    string = token.string.strip()
                elif label[0] == 'I':
                    end = token.idx + len(token.string.strip())
                    string += ' '+token.string.strip()
    # end of the loop
    if start !=-1:
        if entity is not None:
            if form == 'brat':
                new_ann = 'T'+str(idx)+'\t'+entity+' '+str(start)+' '+str(end)+'\t'+string+'\n'
            elif form == 'min':
                new_ann = entity+';'+str(start)+';'+str(end)+'\n'
            annotations.append(new_ann)
            idx+=1
            if visu:
                entities.append({'start':start, 'end':end, 'label':entity})
    return annotations, entities
                
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
