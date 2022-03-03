from torch.distributions import Normal
import torch
import matplotlib.pyplot as plt
import numpy as np

def sample_vae(hid_size, model):
    z = Normal(torch.zeros(hid_size), torch.zeros(hid_size) + 1).sample()
    mu = model.reconstruction_mu(model.decode(z))
    return mu


#

def marginal_KL(valid_loader, model):
    KL = 0
    for X_batch in valid_loader:
        X_batch = X_batch.reshape(valid_loader.batch_size, -1)
        rec_mu, rec_logsigma, latent_mu, latent_logsigma = model(X_batch)
        KL += torch.sum(kl_divergence(latent_mu, latent_logsigma))
    return KL / (len(valid_loader) * valid_loader.batch_size)


def Compute_NLL(valid_loader, model):
    rec_loss = 0
    for X_batch in valid_loader:
        X_batch = X_batch.reshape(valid_loader.batch_size, -1)
        rec_mu, rec_logsigma, latent_mu, latent_logsigma, _ = model(X_batch)
        rec_loss += reconstruction_error(X_batch, rec_mu, 0)
    return rec_loss / (len(valid_loader) * valid_loader.batch_size)


def kl_divergence(mu, log_sigma):
    """
    calculate the kl divergence
    """
    return - 0.5 * torch.sum(1 + 2 * log_sigma - mu.pow(2) - log_sigma.exp().pow(2), dim=1)


def log_likelihood(x, mu, log_sigma):
    """
    calculate the log likelihood, assuming the sigma matrix is a constant multiply by the identity matrix
    """
    return torch.sum(- log_sigma - 0.5 * np.log(2 * np.pi) - (mu - x).pow(2) / (2 * log_sigma.exp().pow(2)), dim=1)


def reconstruction_error(x, x_hat, tol):
    """
    calculate reconstruction error
    """
    return torch.sum(torch.pow(x_hat - x, 2), dim=1) - tol ** 2


def RE_mtr(x, mu, tol):
    return torch.sum(torch.pow(mu - x, 2), dim=(1, 2)) - tol ** 2


def loss_beta_vae(x, mu_latent, logsigma_latent, beta=1):
    return torch.mean(beta * kl_divergence(mu_latent, logsigma_latent) - log_likelihood(x, mu_latent, logsigma_latent))


def draw_hist(train_hist, valid_hist, lambd_hist=[0]):
    fig, ax = plt.subplots(ncols=4, nrows=1, figsize=(16, 4))

    ax[0].plot(train_hist['loss'], label='train')
    ax[0].plot(valid_hist['loss'], label='test')
    ax[0].legend()
    ax[0].set_title("Total Loss")

    ax[1].plot(train_hist['reconstr'], label='train')
    ax[1].plot(valid_hist['reconstr'], label='test')
    ax[1].legend()
    ax[1].set_title("Reconstruction")

    ax[2].plot(train_hist['KL'], label='train')
    ax[2].plot(valid_hist['KL'], label='test')
    ax[2].legend()
    ax[2].set_title("KL divergence")

    ax[3].plot(lambd_hist)
    ax[3].set_title("Lambda")
    plt.show()
