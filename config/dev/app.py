# -*- coding: utf-8 -*-
#    Copyright (C) 2019-TODAY Cleareye.ai(<https://www.cleareye.ai>)
#    Author: Cleareye.ai(<https://www.cleareye.ai>)

import os
from config.common import DefaultConfig


class Development(DefaultConfig):
    def __init__(self):
        self.ce_host = os.environ.get('CE_PREDICT_SKILL_HOST')
        self.ce_port = int(os.environ.get('CE_PREDICT_SKILL_PORT'))

        # model parameters
        self.model_path = os.environ.get('CE_PREDICT_SKILL_MODEL_PATH')

        # log parameters
        self.log_path = os.environ.get('CE_PREDICT_SKILL_LOG_PATH')
        self.log_level = os.environ.get('CE_PREDICT_SKILL_LOG_LEVEL')
        self.log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
