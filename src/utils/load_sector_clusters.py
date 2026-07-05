#!/usr/bin/env python3
"""
Helper script to load and process finance sector allocations from IATI data.
Creates sector cluster features using embeddings and clustering.

Returns DataFrame with:
- sector_hhi: Herfindahl-Hirschman Index (spending concentration)
- sector_cluster_{cluster_name}: allocation % for each cluster
- sector_cluster_{special_sector}: direct allocation for special non-clustered sectors
- n_sectors: number of sectors per activity
"""
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd


# Special sectors that get their own columns and are NOT embedded/clustered
SPECIAL_SECTORS = [
    'increased food production',
    # 'reduced PM2.5 air pollution',
    # 'more people with access to electricity',
]



