from app import app
from flask_cors import CORS
from mangum import Mangum

def handler(event, context):
    CORS(app)
    return Mangum(app)(event, context)