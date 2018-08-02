# Call to the api 
import os
import requests
import optparse
import json
from time import time

# parsing arguments
parser = optparse.OptionParser()
parser.add_option('-i', '--input', help='Input folder of input file')
parser.add_option('-o', '--output', default='output',help='Output folder')
parser.add_option('-v', '--visu', default=False, action='store_true', help='Save a html visualization for each text(default: do not create visualization)')
option, args = parser.parse_args()
if option.input is None:
    print('Input file or directory is required')
    exit(0)
vis_str = 'with' if option.visu else 'without'
print('Getting texts in {} and storing results in {} {} html visualizations'.format(option.input, option.output, vis_str))
input_dir = option.input
output_dir = option.output
VISU_SAVE = option.visu
#input_dir = 'prod_data/inputs/'
#output_dir = 'prod_data/out/'

# api url
TORCH_REST_API_URL = "http://localhost:5000/predict"
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
    print("Created output directory : ", output_dir)

# config 
conf = json.dumps({"visu":VISU_SAVE})

# Get the texts
path2texts = []
if os.path.isdir(input_dir):
    for f in os.listdir(input_dir):
        path2texts.append(os.path.join(input_dir, f))
# case the input dir is one file only 
else:
    path2texts.append(input_dir)

start = time()
print('Decoding {} texts files...'.format(len(path2texts)))
for path2txt in path2texts:
    print('processing file {} ...'.format(path2txt))
    base_name = os.path.splitext(os.path.basename(path2txt))[0]
    with open(path2txt, 'r') as f:
     text = f.read()
    payload = {'file':text, 'filename':base_name, 'conf':conf}
    # api processing
    r = requests.post(TORCH_REST_API_URL, files = payload).json()
    input_data = r['input_data']
    annotations = r['annotations']
    html = r['html']
    # save results
    path2txt = output_dir + base_name + '.txt'
    path2ann = output_dir + base_name + '.ann'
    path2html = output_dir + base_name + '.html'
    if html is not None:
        with open(path2html, 'w', encoding ='utf-8') as f:
            f.write(html)
    with open(path2txt, 'w') as f:
        f.write(input_data)
    with open(path2ann, 'w') as f:
        f.writelines(annotations)

timed = time() - start
print('\n' + '**'*10 + 'Decoding done!' + '**'*10)
print('Total time: {:.4}s'.format(timed))