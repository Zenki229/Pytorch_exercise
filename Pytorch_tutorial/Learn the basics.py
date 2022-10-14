from lib import *

# search the GPU number in machine
# print(torch.cuda.device_count())
#
# # return the gpu name from index 0
# print(torch.cuda.get_device_name([0, 1, 2, 3]))
#
# # return index of current gpu
# print(torch.cuda.current_device())
#
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# tensor = torch.ones((4, 4), device=device)
# print(f"Device tensor is stored in {tensor.device}")
# tensor[:, 1] = 0
# print(tensor @ tensor.T)

''' Bridge with Numpy
    Tensors on the CPU and NumPy arrays can share their underlying memory locations, 
    and changing one will change the other.
'''
t = torch.ones(5, device=device)
print(f"t: {t}\n")
n = t.cpu().numpy()  # note that tensor in gpu should be converted to cpu then numpy
print(f"n: {n}\n")
t.add_(5)
print(f"t: {t}\n")
n = t.cpu().numpy()
print(f"n: {n}\n")

# Numpy array to Tensor
n = np.ones(5)
t = torch.from_numpy(n).to(device)
print(f"t: {t}\n")
print(f"n: {n}\n")
# Changes in the Npy array reflects in the tensor
np.add(n, 1, out=n)
print(f"t: {t}\n")
print(f"n: {n}\n")
