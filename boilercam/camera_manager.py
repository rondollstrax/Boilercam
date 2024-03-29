import base64
import rtsp
import time
import sys

from config import get_config


class CameraManager:
    def __init__(self):

        config = get_config()["camera"]

        pwd = base64.b64decode(config["pass"]).decode('utf-8')
        self.url = f"rtsp://{config['user']}:{pwd}@{config['addr']}:{config['port']}/{config['stream']}"

        self.zoom_hori_start = config["zoom_cube"]["hori_start"]
        self.zoom_vert_start = config["zoom_cube"]["vert_start"]
        self.zoom_length = config["zoom_cube"]["length"]

        self.client = self.init_client()

    def init_client(self):
        client = rtsp.Client(rtsp_server_uri=self.url, verbose=True)
        sanity_start = time.time()
        while client.read() is None:
            time.sleep(1)
            if time.time() - sanity_start > 10:
                print("Camera sanity check failed - terminating")
                sys.exit(1)
        return client
    
    def snip(self):
        img = self.client.read().crop((self.zoom_hori_start, self.zoom_vert_start, self.zoom_hori_start + self.zoom_length, self.zoom_vert_start + self.zoom_length))
        return img


