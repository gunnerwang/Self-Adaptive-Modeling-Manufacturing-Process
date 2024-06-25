import torch
from . import soft_dtw
from . import path_soft_dtw

def dilate_loss(outputs, targets, alpha=0.5, gamma=1e-2, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
	outputs = outputs.unsqueeze(-1)
	targets = targets.unsqueeze(-1)
	outputs, targets = outputs[:, 1:, :], targets[:, 1:, :]
	# outputs, targets: shape (batch_size, N_output, 1)
	batch_size, N_output = outputs.shape[0:2]
	loss_shape = 0
	softdtw_batch = soft_dtw.SoftDTWBatch.apply
	D = torch.zeros((batch_size, N_output,N_output )).to(device)
	for k in range(batch_size):
		Dk = soft_dtw.pairwise_distances(targets[k,:,:].view(-1,1),outputs[k,:,:].view(-1,1))
		D[k:k+1,:,:] = Dk     
	loss_shape = softdtw_batch(D,gamma)
	
	path_dtw = path_soft_dtw.PathDTWBatch.apply
	path = path_dtw(D,gamma)           
	Omega =  soft_dtw.pairwise_distances(torch.arange(1,N_output+1).view(N_output,1)).to(device)
	loss_temporal =  torch.sum( path*Omega ) / (N_output*N_output) 
	loss = alpha*loss_shape + (1-alpha)*loss_temporal
	return loss.unsqueeze(0)  # , loss_shape, loss_temporal
