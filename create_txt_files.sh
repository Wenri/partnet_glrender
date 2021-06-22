#!/bin/bash

cd /cyber/project/partnet/partnet_data
export PYTHONPATH=$(pwd)

python partrender/partnetgroup.py  > /cyber/project/allexports/$1/idname.txt
python partrender/partnetgroup.py | cut -f3 | sort -u > /cyber/project/allexports/$1/grouping.txt
