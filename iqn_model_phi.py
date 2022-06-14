
import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear( 8, 50),
                      nn.ReLU(),
                      
                      nn.Linear(50, 50),
                      nn.ReLU(),
                      
                      nn.Linear(50, 50),
                      nn.ReLU(), 
 
                      nn.Linear(50, 50),
                      nn.ReLU(), 
 
                      nn.Linear(50, 1)) 
