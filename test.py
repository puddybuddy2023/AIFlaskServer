from flask import Flask
from modules.petsnal_colors import *

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'hello'

@app.route('/url_image')
def url_image_test():
    image = process_image_from_url("https://puddybuddybucket.s3.amazonaws.com/images/ac6d4434-ac3a-435b-b8aa-31a5989da144_%E1%84%80%E1%85%B3%E1%84%85%E1%85%B5%E1%86%B71.png-1.png")
    return "good"

if __name__ == '__main__':
    app.run(debug=True)