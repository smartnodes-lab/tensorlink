"""
This will test the request and submission of valid and invalid proof of works between both
worker-worker and worker-validator checks.
"""
from transformers import BertForSequenceClassification
from tensorlink import UserNode
from torch.nn.functional import mse_loss
import logging
import torch
