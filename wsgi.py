import re
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from ner import Parser
from config.default import APP_CONFIG

application = Flask(__name__)
cors = CORS(application)
# extract skill name from the  predicted 'B' and 'I' tags.
# Load parser
# loc = "models/"
skill_parser = Parser(APP_CONFIG.model_path)


@application.route('/api/predict', methods=['POST'])
def skill_predict():

    try:
        start_time = time.time()
        input_dict = request.get_json()
        print("input_dict", input_dict)
        sentence = input_dict['text']

        output = ""
        if sentence:
            output = skill_parser.predict(sentence)  # call the predict function for each sentence

        print("total time taken", time.time() - start_time)
    except Exception as e:
        output = {
            'status': 'error',
            'code': e,
        }
    return jsonify(output)


if __name__ == '__main__':
    application.run(debug=True, threaded=True, host=APP_CONFIG.ce_host, port=APP_CONFIG.ce_port)
