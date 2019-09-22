from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
import os
import torch
from net import *
from graph import get_graph


app = Flask(__name__)

Bootstrap(app)

UPLOAD_DIR = '/home/whale/PycharmProjects/gazprom/uploaded_data/'
MODEL_PATH = '/home/whale/PycharmProjects/gazprom/static/cnn_model.dms'

model = Net()
model.load_state_dict(torch.load(MODEL_PATH))


@app.route('/', methods=['GET', 'POST'])
def welcome():

    # model = Net()
    # model.load_state_dict(torch.load(MODEL_PATH))
    # torch.no_grad()
    # model.eval()

    if request.method == 'POST':
        try:
            file = request.files['file']
        except:
            return render_template('welcome_page.html')
        if file:
            filename = file.filename

            file.save(os.path.join(UPLOAD_DIR, filename))
            #render graph
            graph_name = get_graph(UPLOAD_DIR + filename, model)

            return render_template('prediction.html', graph=graph_name)
    return render_template('welcome_page.html', title='Application')


if __name__ == '__main__':

    app.run(debug=True)
