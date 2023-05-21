#! /usr/bin/python

import time
import os
import json
import requests

from pathlib import Path

from camera_manager import CameraManager
# from image_classifier import ImageClassifier
from fastai_classifier import FastAIClassifier
from config import get_config

class BoilerUpdater:
    def __init__(self):
        config = get_config()['site_host']
        self.port = config['port']
        self.addr = f"{config['scheme']}://{config['addr']}"
        self.auth = config['auth_token']

    def send(self, is_on):
        body = {"on": "1" if is_on else "0"}
        headers = {"Authorization": self.auth}
        try:
            requests.post(f'{self.addr}:{self.port}', body, headers=headers)
        except requests.exceptions.RequestException as e:
            print('Request failed with {}'.format(e))
            return
        print('sent! {}'.format(body))

class BoilerCam:
    def __init__(self):
        self.json_file = os.path.join(Path(__file__).parent, 'site', 'boiler.json')
        self.is_the_boiler_on = self.read_json()['isTheBoilerOn']
        self.camera_manager = CameraManager()
        #self.image_classifier = ImageClassifier()
        self.fast_ai_classifier = FastAIClassifier()
        self.boiler_updater = BoilerUpdater()

    def read_json(self):
        with open(self.json_file, 'r') as jsonfile:
            j = json.load(jsonfile)
        return j

    def write_boiler_state(self, is_on):
        self.is_the_boiler_on = is_on
        with open(self.json_file, 'w') as jsonfile:
            json.dump({'isTheBoilerOn': is_on}, jsonfile)

    def run(self):
        image = self.camera_manager.snip()
        #is_on = bool(self.image_classifier.predict(image))
        is_on = bool(self.fast_ai_classifier.predict(image))
        print('is_on is', is_on)
        print('comparing')
        if is_on != self.is_the_boiler_on:
            print('writing locally')
            self.write_boiler_state(is_on)
            print('updating remotely')
            self.boiler_updater.send(is_on)
            print('sent')

if __name__ == '__main__':
    boiler_cam = BoilerCam()
    while 1:
        boiler_cam.run()
        time.sleep(5)
        
