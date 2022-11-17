#!/usr/bin/env python
## Youcef Chorfi
import yaml
from pathlib import Path

def Read_yaml() :
    path = str(Path(__file__).parent.parent.parent)+'/config/params.yaml'
    with open(path, 'r') as f:
        data = yaml.full_load(f)
    return data