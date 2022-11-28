#!/usr/bin/env python
## Youcef Chorfi
import yaml
from pathlib import Path


def getParams() :
    path = str(Path(__file__).parent.parent.parent)+'/config/params.yaml'
    try :
        with open(path, 'r') as f:
            data = yaml.full_load(f)
    except:
        raise ValueError("error opening params.yaml file")

    return data