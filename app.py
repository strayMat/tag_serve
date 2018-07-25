import os
import sys
import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import spacy
from spacy import displacy

from serve import get_model_api # class to implement which loads the model and has predict and/or train functions

#!!!!!TODO add argument like model archi and weights paths and verbosity (default paths would be pretrained light file)
VISU = True
VISU_SAVE = True
OUT_PATH = 'processed/'

# define the app
app = Flask(__name__)
CORS(app) # cross-domain requests, allow everything by default 

# loading model once and for all the api
model_api = get_model_api()

# loading tokenizer once and for all for the api
app.logger.info('Loading spacy tokenizer...')
nlp = spacy.load('en', disable=['tagger', 'ner', 'parser'])
nlp.add_pipe(nlp.create_pipe('sentencizer')) 

#STATUS = 'live' # live/file
#path2docs = 'prod_data/wiki_en_france.txt'

# API live demo route
@app.route('/api', methods=['POST'])
def api():
    input_data = request.json
    app.logger.info('api_input: ' + str(input_data))
 	
    input_client, output_client = model_api(input_data, tokenizer = nlp)
    app.logger.info('api_output: ' + str(output_client))
    response = jsonify(input= input_client, output = output_client)
    return response

# API live predict 
## commandline
# curl -X POST -F file=@prod_data/wiki_en_france.txt 'http://localhost:5000/file_predict'
@app.route('/file_predict', methods =['POST'])
def file_api():
    # upload a file
    #file = '../prod_data/wiki_en_france.txt'
    if request.method == 'POST':
        if request.files.get("file"):
            
            # read the image in PIL format
            input_data = request.files["file"].read().decode('utf-8')
            filename = request.files["file"].filename
            app.logger.info('api input file:' + str(filename))
            
            # output folder 
            out_folder = OUT_PATH
            if not os.path.isdir(out_folder):
                os.mkdir(out_folder)
            path2txt = out_folder + os.path.splitext(filename)[0] + '.txt'
            path2ann = out_folder + os.path.splitext(filename)[0] + '.ann'
            path2html = out_folder + os.path.splitext(filename)[0] + '.html'
    
            app.logger.info('api_input: ' + str(input_data))
            
            # Predict
            input_client, output_client = model_api(input_data, live = False, tokenizer = nlp)
            app.logger.info('api_output: ' + str(output_client))
            # post processing of the data:
            out = [' '.join(sent) + '\n' for sent in output_client]

            if VISU:
                entities = []
            annotations = []
            idx = 1
            text = nlp(input_data)
            for sent, label_seq in zip(text.sents, output_client):
                for token, label in zip(sent, label_seq):
                    if label != "O":
                        new_ann = 'T'+str(idx)+'\t'+label[2:]+' '+str(token.idx)+' '+str(token.idx+len(token.string.strip()))+'\t'+token.string.strip()+'\n'
                        annotations.append(new_ann)
                        idx+=1
                        if VISU:
                            entities.append({'start':token.idx, 'end':token.idx+len(token.string.strip()), 'label':label[2:]})
            
            # write files
            with open(path2txt, 'w') as f:
                f.write(input_data)
            with open(path2ann, 'w') as f:
                f.writelines(annotations)

            if VISU:
                visu_data = [{'text':input_data, 'ents':entities, 'title':None}]
                if VISU_SAVE:
                    html = displacy.render(visu_data, style='ent', page=True, manual=True)
                    with open(path2html, 'w', encoding ='utf-8') as f:
                        f.write(html)
                displacy.serve(visu_data, style='ent', manual=True, port=5001)

            ## !!!!!!!!!!!!!!what is the response ??? do we want to return the file and how ?? !!!!!!!!!!!!
            response = jsonify(input= input_data, output = out)

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
