import torch
import torch.nn as nn
import pandas as pd

class data_create:
    def __init__(self,path_asv,path_chem):
        self.asv_data = pd.read_csv(path_asv)
        self.chem_data = pd.read_excel(path_chem)
    def __iter__(self):
        yield self.asv_data
        yield self.chem_data

asv,chem= data_create('data\\raw\\taxon_data\\taxon_lv7.csv','data\\raw\\chem_data.xlsx')
print(asv)
print(chem)
