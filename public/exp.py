from flask import Flask
from flask import request

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/<noun>")
def print(noun:str):
    return f"<p>A Noun: {noun}</p>"

@app.route("/plus", methods=['POST'])
def plus():
    num_dict = request.get_json()
    _sum = 0
    for i in num_dict.values():
        _sum += i
    return str(_sum)

if __name__ == '__main__':
    app.run(host='192.168.1.101', port=300, debug=False)