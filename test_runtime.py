import torch
import sys
sys.path.append("./src")
from LPTN_Network import LPTN_Network
LPTN=LPTN_Network().to('cuda')
LPTN.load_state_dict(torch.load("Experiments/2024-04-11_02-20-36/Checkpoints/425.pth"))
sizes=[(720,480), (1920, 1080), (2560,1440), (3840,2160)]
num_iterations=50
times=[0,0,0,0]
for i,size in enumerate(sizes):
    random_image= torch.rand((1,3,size[0],size[1])).to('cuda')    
    for q in range(num_iterations):
        _=LPTN(random_image)
        times[i]+=LPTN.duration
    
print([time/num_iterations for time in times])