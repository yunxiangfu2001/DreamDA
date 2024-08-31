import torch
import torch.nn as nn
from kmeans_pytorch import kmeans

def generate_diverse_nosie(size,num_noise=10, generator=None, dtype=torch.float16, device='cpu'):
    """
    Generate a noise tensor with the same shape as the given tensor.
    :param size: shape of the tensor to generate noise for
    :param generator: torch.Generator to use for noise generation
    :param dtype: type of the tensor to generate noise for
    :param device: device to generate noise on
    :return: tensor with the same shape as the given tensor, but with noise
    """
    if generator is None:
        generator = torch.Generator(device).manual_seed(123456789)

    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    
    noises = [torch.randn(size, device=device) for _ in range(num_noise*100)]
    noises = torch.stack([noise.view(-1) for noise in noises])

    
    diverse_noise = noises[:num_noise]
    
    similarity_matrix = cos(diverse_noise[None,:,:],diverse_noise[:,None,:])
    similarity_matrix = similarity_matrix - torch.eye(similarity_matrix.size(0),device=device)
    for noise in noises[num_noise:]:
        each_diversity = cos(diverse_noise, noise.unsqueeze(0))

        
        values, indices = torch.max(similarity_matrix.sum(dim=1),dim=0)
        each_diversity[indices] = torch.zeros(1)

        if each_diversity.sum() < values.sum():
            similarity_matrix[indices] = each_diversity
            diverse_noise[indices] = noise


    return noise
