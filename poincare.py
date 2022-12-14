#!/usr/bin/env python3

from aipoincare.main import run

from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

if __name__ == '__main__':
    app.run(host="localhost", port=8000)
    run()
