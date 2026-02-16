#!/usr/bin/env python3
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import uinput

device = None

class MouseControlHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        try:
            data = json.loads(post_data)
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'Bad Request: Invalid JSON')
            return

        x = data.get('x', 0)
        y = data.get('y', 0)

        if not isinstance(x, int) or not isinstance(y, int):
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'Bad Request: x and y must be integers')
            return

        print(x, y)
        device.emit(uinput.REL_X, x, syn=False)
        device.emit(uinput.REL_Y, y)

        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'OK')

def main():
    events = (
        uinput.REL_X,
        uinput.REL_Y,
        uinput.BTN_LEFT,
        uinput.BTN_RIGHT,
    )

    global device
    # The script must be run as root to have permission for uinput
    with uinput.Device(events) as dev:
        device = dev
        server_address = ('localhost', 8000)
        httpd = HTTPServer(server_address, MouseControlHandler)
        print(f'Starting server on {server_address[0]}:{server_address[1]}...')
        httpd.serve_forever()

if __name__ == "__main__":
    main()
