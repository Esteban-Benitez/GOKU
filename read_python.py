import pickle
import torch
with open('checkpoints/pendulum/goku_model.pkl', 'rb') as file:
    data = torch.load(file, map_location='cpu', weights_only=False)

#print(data)
print(f"args: {data['args']}")
print("-----------------------------")
print(f"model: {data['model']}")

    
