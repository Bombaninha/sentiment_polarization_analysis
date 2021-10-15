<<<<<<< HEAD
from flask import Flask
from flask_restful import Resource, Api
from app import *
import json

#print(analyze("amor paixão apaixonado criança felicidade"))

app = Flask(__name__)
api = Api(app)

class ModelAnalyzer(Resource):
    def get(self, phrase):
        return { 'data' : json.dumps(analyze(phrase), default=lambda x: x.__dict__) }

prefix = '/api/v1/'
api.add_resource(ModelAnalyzer, prefix + 'analyze/<string:phrase>')

if __name__ == '__main__':
    app.run(debug=True)
=======
from flask import Flask
from flask_restful import Resource, Api
from app import *
import json

#print(analyze("amor paixão apaixonado criança felicidade"))

app = Flask(__name__)
api = Api(app)

class ModelAnalyzer(Resource):
    def get(self, phrase):
        return { 'data' : json.dumps(analyze(phrase), default=lambda x: x.__dict__) }

prefix = '/api/v1/'
api.add_resource(ModelAnalyzer, prefix + 'analyze/<string:phrase>')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
>>>>>>> main
