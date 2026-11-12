# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 09:37:39 2026

@author: david
"""

import urllib.request
import os

def download_text8(url, target_folder):
    os.makedirs(target_folder, exist_ok=True)
    target_file = os.path.join(target_folder, 'text8.zip')

    if not os.path.isfile(target_file):
        #prvo se skida u privremeni fajl, da prekinuto preuzimanje ne ostavi
        #nepotpun text8.zip koji bi se sledeci put preskocio
        privremeni_fajl = target_file + '.part'
        urllib.request.urlretrieve(url, privremeni_fajl)
        os.replace(privremeni_fajl, target_file)


if __name__ == "__main__":
    text8_url = "http://mattmahoney.net/dc/text8.zip"
    target_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "text8_dataset")


    download_text8(text8_url, target_folder)