from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import requests
import zipfile
import os
import sys
from pdb import set_trace as bp
from tqdm import tqdm
import math

data_dict = {
    'pth':'1mmHrBVl16HQEjLpXk--6Z5dwfbselKBW',
    'CASIA_Webface_160':'175YhXe26wMMxSRuKGAbbVCkY5MLDk5m7', 
    'LFW_112': '11-uZAudZsBX5NkmYtYeMR0PPaWN3KVSG',
    'CPLFW_112': '1YeWzDL8XmAWXoRx5mVcObxwczGLP7Tsh',
    'CALFW_112': '1J2KXbbfBpxxFdvPhAi0q5MHjTS7O4Ai6',
    'CFP_112': '10yt5mx8ENx-QGOAXw2bfVUOfgN7rTKMY',
    'dataset_got': '1bV2nb47pgk6EJlUbsAxWFsfJTquseMdJ'
    }

def download_and_extract_file(model_name, data_dir):
    file_id = data_dict[model_name]
    destination = os.path.join(data_dir, model_name + '.zip')
    if not os.path.exists(destination):
        print('Downloading file to %s' % destination)
        download_file_from_google_drive(file_id, destination)
        with zipfile.ZipFile(destination, 'r') as zip_ref:
            print('Extracting file to %s' % data_dir)
            zip_ref.extractall(data_dir)
            print("Removing archive file {}".format(destination))
            if os.path.exists(destination):
                os.remove(destination)
            else:
                print("Failed to remove file {}".format(destination))

def download_file_from_google_drive(file_id, destination):
    
        URL = "https://drive.google.com/uc?export=download"

        session = requests.Session()
    
        print("Downloading %s" % destination)
        headers = {'Range':'bytes=0-'}
        r = session.get(URL,headers=headers, params = { 'id' : file_id }, stream = True)

        token = get_confirm_token(r)
        if token:
            params = { 'id' : file_id, 'confirm' : token }
            r = session.get(URL,headers=headers, params = params, stream = True)
            save_response_content(r, destination)
        else:
            save_response_content(r, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(r, destination):
    rr = r.headers['Content-Range']
    total_length=int(rr.partition('/')[-1])

    block_size = 32768
    wrote = 0 
    with open(destination, 'wb') as f:
        for data in tqdm(r.iter_content(block_size), total=math.ceil(total_length//block_size) , unit='KB', unit_scale=True):
            wrote = wrote  + len(data)
            f.write(data)
    if total_length != 0 and wrote != total_length:
        print("ERROR, something went wrong")  


if __name__ == '__main__':
    
    out_dir = 'data/'
    if not os.path.isdir(out_dir):  # Create the out directory if it doesn't exist
        os.makedirs(out_dir)

    download_and_extract_file('LFW_112', out_dir)
    download_and_extract_file('CPLFW_112', out_dir)
    download_and_extract_file('CALFW_112', out_dir)
    download_and_extract_file('CFP_112', out_dir)

    download_and_extract_file('CASIA_Webface_160', out_dir)

    download_and_extract_file('pth', out_dir)
    download_and_extract_file('dataset_got', out_dir)


