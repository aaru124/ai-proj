from flask import Flask as f
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pandas as pd
import seaborn as sns

def create_app():
    app = f(__name__,static_url_path='/static',static_folder='static')
    
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///db.sqlite3"
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SECRET_KEY'] = "abcd234"


    app.app_context().push()

    return app

app = create_app()

from application.controllers import *

if __name__=="__main__":
    app.run(host='0.0.0.0', port=2080, debug=True)