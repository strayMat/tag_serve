import os
import sys
import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from serve import get_model_api # class to implement which loads the model and has predict and/or train functions

# define the app
app = Flask(__name__)
CORS(app) # cross-domain requests, allow everything by default TODO:WHAAAAT ???
# loading once and for all the api
model_api = get_model_api()

# API route
@app.route('/api', methods =['POST'])
def api():
    input_data = request.json
    app.logger.info('api_input: ' + str(input_data))
 	## log information here like speed of loading or 
    input_client, output_client = model_api(input_data)
    app.logger.info('api_output: ' + str(output_client))
    response = jsonify(input= input_client, output = output_client)
    return response

# default route
@app.route('/')
def index():
    return "Index API: torch_serve is on b$$$$ch!"

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