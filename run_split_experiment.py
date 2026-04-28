from data.eurosat import get_dataloaders
from architectures.model_factory import build_model
from engine.train import train_model
from engine.evaluate import evaluate_model
from utils.config import load_config
import datetime
import os
import json
