#!/home/alex/miniconda3/envs/torch/bin/python

import PyLTSpice as lt
import numpy as np
from config import asc_path

net = lt.AscEditor(asc_path)
net.set_parameter("Vto", 1)
net.save_netlist(asc_path)