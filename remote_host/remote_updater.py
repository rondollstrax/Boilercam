from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import os
from config import get_config

def write_json(on):
    path = os.path.join(Path(__file__).parent, 'site', 'boiler.json')
    with open(path, 'w') as jsonfile:
        json.dump({'isTheBoilerOn': on}, jsonfile)

class S(BaseHTTPRequestHandler):
    def __init__(self):
        super().__init__()
        self.auth = get_config()['auth_token']

    def _set_response(self):
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        if self.headers.getheader('Authorization') != self.auth:
            return

        self._set_response()
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')[-1]
        write_json(True if int(post_data) == 1 else False)


def run(server_class=HTTPServer, handler_class=S, port=56565):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()

if __name__ == '__main__':
    run()
