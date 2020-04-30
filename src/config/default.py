# -*- coding: utf-8 -*-
#    Copyright (C) 2019-TODAY Cleareye.ai(<https://www.cleareye.ai>)
#    Author: Cleareye.ai(<https://www.cleareye.ai>)
import os
from config.dev.app import Development
from config.prod.app import ProductionConfig

# Load the development "mode". Use "development" if not specified
ENV = os.environ.get("CE_APP_ENV", "development")


# Configuration for each environment
# Alternatively use "python-dotenv"
def get_env(ENV):
    if ENV == "development":
        return Development()
    elif ENV == "production":
        return ProductionConfig()


# The config for the current environment
APP_CONFIG = get_env(ENV)
