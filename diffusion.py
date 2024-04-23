import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import matplotlib.pyplot as plt

class Diffusion(nn.Module):
  def __init__(self, time = 1000, images_size = (64, 64), dtype = torch.float64, device = "cpu"):
    super(Diffusion, self).__init__()
    self.T = time
    self.device = device
    self.dtype = dtype
    self.images_size = images_size
    self.alpha_hat = self.get_alpha_hat()
    self.beta = self.get_beta()
    # self.alpha_hat = self.alpha_hat[1:]

  def get_beta(self):
    return torch.linspace(1e-4, 0.02, self.T, dtype = self.dtype, device = self.device)

  def get_alpha_hat(self):
    alpha = 1 - torch.linspace(1e-4, 0.02, self.T, dtype = self.dtype, device = self.device)
    alpha_hat = torch.cumprod(alpha, dim=0)
    return alpha_hat

  # calculate q(x_t | x_0) = sqrt(alpha_hat[t]) * x[0] + sqrt(1 - alpha_hat[t]) * N(0, 1)
  # return both x_t and noise
  def get_noise(self, x, t):
    alpha_hat_sqrt = torch.sqrt(self.alpha_hat[t]).reshape(-1, 1, 1, 1)
    one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t]).reshape(-1, 1, 1, 1)
    error = torch.randn_like(x)
    return alpha_hat_sqrt * x + one_minus_alpha_hat * error, error

  # return the loss of the model
  def forward(self, model, images):
    # sample t uniformly for each image in batch
    t = torch.randint(self.T, size = (images.shape[0],), dtype = torch.long, device = self.device)
    x_t, noise = self.get_noise(images, t)
    predicted_noise = model(x_t, t + 1)
    return F.mse_loss(predicted_noise, noise)

  # implement the sampling algorithm from https://arxiv.org/pdf/2006.11239.pdf
  def inference(self, model, noise, toImage = False):
    model.eval()
    with torch.no_grad():
      x = noise
      for i in range(self.T, 0, -1):
        z = torch.randn_like(x)
        if i == 1:
          z *= 0.0
          
        t = (torch.ones(noise.shape[0], dtype = torch.long) * i).to(self.device)

        alpha = 1 - self.beta[i - 1]
        alpha_hat = self.alpha_hat[i - 1]
        predicted_noise = model(x, t)

        x = 1.0 / torch.sqrt(alpha) * (x - ((1.0 - alpha) / torch.sqrt(1.0 - alpha_hat)) * predicted_noise) \
          + torch.sqrt(self.beta[i - 1]) * z

    # convert back to range [-1, 1]
    x = x.clamp(-1, 1)

    # convert back to image [0, 255]
    if toImage:
      x = (x + 1) / 2
      x = (x * 255).type(torch.uint8)
    return x

class ImprovedDiffusion(Diffusion):
  def __init__(self, time = 1000, images_size = (64, 64), dtype = torch.float64, device = "cpu"):
    super(ImprovedDiffusion, self).__init__(time, images_size, dtype, device)
    self.alpha_hat = self.alpha_hat[1:]
    
  def get_beta(self):
    beta = self.alpha_hat[1:] / self.alpha_hat[:-1]
    return (1 - beta).clamp(max = 0.999)

  def get_alpha_hat(self):
    s = 0.008
    t = torch.arange(self.T + 1, dtype = self.dtype, device = self.device) / self.T
    alpha_hat = torch.cos(((t + s)/ (1 + s)) * (math.pi / 2.0)) ** 2
    alpha_hat[1:] /= alpha_hat[0]   
    
    # start = 1.0 
    # end= 0.01
    # alpha_hat = torch.linspace(start, end, steps=self.T, self.dtype, device = self.device)
    return alpha_hat

class Improved_CFG_Diffusion(Diffusion):
  def __init__(self, time = 1000, images_size = (64, 64), dtype = torch.float64, device = "cpu"):
    super(Improved_CFG_Diffusion, self).__init__(time, images_size, dtype, device)

  # implement the CFG sampling algorithm from https://arxiv.org/pdf/2207.12598.pdf
  def inference(self, model, noise, guidance_strength = 3, label = None, logging = False):
    model.eval()
    with torch.no_grad():
      x = noise
      for i in range(self.T, 0, -1):
        z = torch.randn_like(x)
        # if i == 1:
        #   z *= 0.0
        t = (torch.ones(noise.shape[0], dtype = torch.long) * i).to(self.device)

        predicted_noise = model(x, t, label)
        if guidance_strength > 0:
            uncond_predicted_noise = model(x, t, None)
            predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, guidance_strength + 1)
            
        alpha = 1 - self.beta[i - 1]
        alpha_hat = self.alpha_hat[i - 1]
        beta = self.beta[i - 1]

        x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat + 1e-6))) * predicted_noise) 
        if i != 1:
          x += torch.sqrt(beta) * z
        if logging == True and i % 100 == 0:
            plt.imshow(((x[0].clamp(-1, 1) + 1) / 2).cpu().permute(1, 2, 0))
            plt.show()
  
    model.train()
    # convert back to range [0, 1]
    x = (x.clamp(-1, 1) + 1) / 2 
    return x

  def compressed_inference(self, model, noise, guidance_strength = 3, label = None, logging = False, sequence_length = 50):
    seq = torch.linspace(1, self.T, sequence_length, dtype = self.dtype, device = self.device)
    seq = torch.round(seq).to(dtype = torch.int)
    # print(seq)
    model.eval()
    with torch.no_grad():
      x = noise
      for u in range(len(seq) - 1, -1, -1):
        i = seq[u]

        z = torch.randn_like(x)
        # if i == 1:
        #   z *= 0.0
        t = (torch.ones(noise.shape[0], dtype = torch.long).to(self.device) * i).to(self.device)

        predicted_noise = model(x, t, label)
        if guidance_strength > 0:
            uncond_predicted_noise = model(x, t, None)
            predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, guidance_strength + 1)
            
        alpha_hat = self.alpha_hat[i - 1]
        
        beta = alpha_hat
        if u != 0:
          prev = seq[u - 1]
          beta /= self.alpha_hat[prev - 1]

        beta = 1 - beta
        alpha = 1 - beta

        x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat + 1e-6))) * predicted_noise) 
        
        if u != 0:
          x += torch.sqrt(beta) * z

        if logging == True and (u + 1) % 20 == 0:
            plt.imshow(((x[0].clamp(-1, 1) + 1) / 2).cpu().permute(1, 2, 0))
            plt.show()
  
    model.train()
    # convert back to range [0, 1]
    x = (x.clamp(-1, 1) + 1) / 2 
    return x

  def inference_gif(self, model, noise, guidance_strength = 3, label = None):
    model.eval()
    ans = []
    with torch.no_grad():
      x = noise
      for i in range(self.T, 0, -1):
        z = torch.randn_like(x)
        # if i == 1:
        #   z *= 0.0
        t = (torch.ones(noise.shape[0], dtype = torch.long) * i).to(self.device)

        predicted_noise = model(x, t, label)
        if guidance_strength > 0:
            uncond_predicted_noise = model(x, t, None)
            predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, guidance_strength + 1)
            
        alpha = 1 - self.beta[i - 1]
        alpha_hat = self.alpha_hat[i - 1]
        beta = self.beta[i - 1]

        x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat + 1e-6))) * predicted_noise) 
        if i != 1:
          x += torch.sqrt(beta) * z
        if i % 10 == 0:
            ans.append(((x.clamp(-1, 1) + 1) / 2)[0].permute(2, 1, 0))
  
    model.train()
    # convert back to range [0, 1]
    x = (x.clamp(-1, 1) + 1) / 2 
    ans.append(x[0].permute(2, 1, 0))
    return ans

  # implement the CFG training algorithm from https://arxiv.org/pdf/2207.12598.pdf
  def forward(self, model, images, labels = None):
    # sample t uniformly for each image in batch
    t = torch.randint(self.T, size = (images.shape[0],), dtype = torch.long, device = self.device)
    x_t, noise = self.get_noise(images, t)
    predicted_noise = model(x_t, t + 1, labels)
    return F.mse_loss(predicted_noise, noise)




