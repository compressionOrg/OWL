import numpy as np
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from importlib.metadata import version
from transformers import AdamW
from datasets import load_dataset
from .data import get_loaders
import torch.nn as nn 
from tqdm import tqdm
import argparse
import os
import json
from pdb import set_trace as st 
from .utils import find_layers

class Layer:
    def __init__(self, model):
        self.model = model
        self.node = dict()
        self.conn_l1 = dict()
        self.conn_l2 = dict()
        self.total_conn = dict()
        self.total_node = dict()
        self.nsample = 0
        self.device = torch.device("cpu") 
        self.layer_init()

    def layer_init(self):
        layers = self.model.model.layers
        for i in tqdm(range(len(layers)), desc=f"initializing the gradient list ...."):
            layer = layers[i]
            subset = find_layers(layer)
            for name in subset:
                indexed_name = f"{name}_layer_{i}"
                self.node[indexed_name] = torch.zeros_like(subset[name].weight, dtype=torch.float32, device=self.device)
                # self.conn_l1[indexed_name] = torch.zeros_like(subset[name].weight, dtype=torch.float16, device=self.device)
                self.conn_l2[indexed_name] = torch.zeros_like(subset[name].weight, dtype=torch.float32, device=self.device)
    
    def update_layer(self, model, nsample):
        assert nsample - self.nsample == 1, "number of samples must be incremented by 1"
        layers = model.model.layers
        for i in tqdm(range(len(layers)), desc=f"updating the gradient of sample no: {self.nsample}"):
            layer = layers[i]
            subset = find_layers(layer)
            per_layer_conn = []
            per_layer_node = []
            for name in subset:
                indexed_name = f"{name}_layer_{i}"
                self.node[indexed_name] = subset[name].weight.data.clone().to(dtype=torch.float32).to(device=self.device)
                per_layer_node.append(self.node[indexed_name])
                if subset[name].weight.grad is None:
                    print(f"Error: {name} has none gradient")
                if subset[name].weight.grad is not None:
                    assert subset[name].weight.requires_grad == True, f"Required grad must be true ( {name}: {subset[name].weight.requires_grad})"
                    grad = subset[name].weight.grad.detach().clone().to(dtype=torch.float32)  # Cast to float32
                    all_zero = (torch.abs(grad)==0).all()
                    assert int(all_zero) == 0, f"all the elements in the tensor are zero.: {all_zero}"
                    # assert self.conn_l1[indexed_name].shape == grad.shape, "shape mismatch"
                    
                    # 去掉scale
                    # self.conn_l1[indexed_name] = self.conn_l1[indexed_name] + torch.abs(grad*self.scale).to(device=self.device).to(dtype=torch.float16)
                    # self.conn_l2[indexed_name] = self.conn_l2[indexed_name] + torch.abs((grad*self.scale)**2).to(device=self.device)
                    
                    # self.conn_l1[indexed_name] = self.conn_l1[indexed_name] + torch.abs(grad).to(device=self.device).to(dtype=torch.float16)
                    self.conn_l2[indexed_name] = self.conn_l2[indexed_name] + torch.abs((grad)**2).to(device=self.device)
                    
                    per_layer_conn.append(self.conn_l2[indexed_name])
                    
            self.total_conn[i] =  torch.cat([torch.flatten(x.cpu()) for x in per_layer_conn])
            self.total_node[i] =  torch.cat([torch.flatten(torch.abs(x.cpu())) for x in per_layer_node])
        self.nsample = nsample