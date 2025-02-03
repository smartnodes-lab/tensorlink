"""
This will test the request and submission of valid and invalid proof of works between both
worker-worker and worker-validator checks.
"""

import logging

import torch
from torch.nn.functional import mse_loss
from transformers import BertForSequenceClassification

from tensorlink import UserNode
