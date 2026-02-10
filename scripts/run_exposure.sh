#!/bin/bash

# time python flood_event_generation_AFR.py C:/Users/cenv1075/Desktop/DataFolders/JBA_flooding/incoming_data/GlobalEventSet/R11/ObsEventRp_20161207.csv C:/Users/cenv1075/Desktop/DataFolders/JBA_flooding

    
python flood_event_generation_AFR.py \
    "soge-home/projects/mistral/JBA_flooding/incoming_data/GlobalEventSet/R11/ObsEventRp_20161207.csv" \
    "soge-home/users/cenv1075/Projects/JBA_flooding"

python flood_event_generation_AFR.py \
    "soge-home/projects/mistral/JBA_flooding/incoming_data/GlobalEventSet/R12/ObsEventRp_20161126.csv" \
    "soge-home/users/cenv1075/Projects/JBA_flooding"

python flood_event_generation_AFR.py \
    "soge-home/projects/mistral/JBA_flooding/incoming_data/GlobalEventSet/R11/SimEventRp_20161207.csv" \
    "soge-home/users/cenv1075/Projects/JBA_flooding"

python flood_event_generation_AFR.py \
    "soge-home/projects/mistral/JBA_flooding/incoming_data/GlobalEventSet/R12/SimEventRp_20161126.csv" \
    "soge-home/users/cenv1075/Projects/JBA_flooding"