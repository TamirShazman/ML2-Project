import torch
import time
from IPython import display
from functions import kl_divergence, log_likelihood, reconstruction_error, RE_mtr, loss_beta_vae, draw_hist
import models


class Train_Geco:
    def __init__(self, model, optimizer, scheduler, train_loader, test_loader, init_lambda=torch.FloatTensor([1]),
                 constraint_f=RE_mtr, num_epochs=20, lbd_step=100, alpha=0.99, verbose=True, device='cpu', tol=1,
                 pretrain=1):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.init_lambda = init_lambda
        self.constrain_f = constraint_f
        self.num_epochs = num_epochs
        self.lbd_step = lbd_step
        self.alpha = alpha
        self.verbose = verbose
        self.device = device
        self.tol = tol
        self.pretrain = pretrain

    def train(self):
        parametrized = self.model.parametrized
        self.model.to(self.device)
        train_hist = {'loss': [], 'reconstr': [], 'KL': []}
        test_hist = {'loss': [], 'reconstr': [], 'KL': []}
        lambd_hist = []

        lambd = self.init_lambda.to(self.device)
        iter_num = 0

        for epoch in range(self.num_epochs):
            start_time = time.time()

            self.model.train(True)
            train_hist['loss'].append(0)
            train_hist['reconstr'].append(0)
            train_hist['KL'].append(0)

            for X_batch in self.train_loader:

                if isinstance(self.model, models.ConvolutionalDRAW):
                    X_batch = X_batch.to(self.device)
                    x_hat, kl, _, _ = self.model(X_batch)
                    constraint = torch.mean(self.constrain_f(X_batch, x_hat, tol=self.tol))
                    kl_div = torch.mean(kl.sum(dim=(1, 2, 3)))
                else:
                    X_batch = X_batch.reshape(self.train_loader.batch_size, -1).to(self.device)
                    if parametrized:
                        reconstruction_mu, reconstruction_logsigma, latent_mu, latent_logsigma, _ = self.model(X_batch)
                        constraint = - torch.mean(self.constrain_f(X_batch, reconstruction_mu, reconstruction_logsigma))
                    else:
                        x_hat, latent_mu, latent_logsigma, _ = self.model(X_batch)
                        constraint = torch.mean(self.constrain_f(X_batch, x_hat, tol=self.tol))
                    kl_div = torch.mean(kl_divergence(latent_mu, latent_logsigma))

                loss = kl_div + lambd * constraint
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                with torch.no_grad():
                    if epoch == 0 and iter_num == 0:
                        constrain_ma = constraint
                    else:
                        constrain_ma = self.alpha * constrain_ma.detach_() + (1 - self.alpha) * constraint
                    if iter_num % self.lbd_step == 0 and epoch > self.pretrain:
                        lambd *= torch.clamp(torch.exp(constrain_ma), 0.95, 1.05)

                train_hist['loss'][-1] += loss.data.cpu().numpy()[0] / len(self.train_loader)
                train_hist['reconstr'][-1] += constraint.data.cpu().numpy() / len(self.train_loader)
                train_hist['KL'][-1] += kl_div.data.cpu().numpy() / len(self.train_loader)
                iter_num += 1
            lambd_hist.append(lambd.data.cpu().numpy()[0])

            self.model.train(False)
            test_hist['loss'].append(0)
            test_hist['reconstr'].append(0)
            test_hist['KL'].append(0)
            with torch.no_grad():
                for X_batch in self.test_loader:

                    if isinstance(self.model, models.ConvolutionalDRAW):
                        X_batch = X_batch.to(self.device)
                        x_hat, kl, _, _ = self.model(X_batch)
                        constraint = torch.mean(self.constrain_f(X_batch, x_hat, tol=self.tol))
                        kl_div = torch.mean(kl.sum(dim=(1, 2, 3)))
                    else:
                        X_batch = X_batch.reshape(self.train_loader.batch_size, -1).to(self.device)
                        if parametrized:
                            reconstruction_mu, reconstruction_logsigma, latent_mu, latent_logsigma, _ = \
                                self.model(X_batch)
                            constraint = - torch.mean(
                                self.constrain_f(X_batch, reconstruction_mu, reconstruction_logsigma))
                        else:
                            x_hat, latent_mu, latent_logsigma, _ = self.model(X_batch)
                            constraint = torch.mean(self.constrain_f(X_batch, x_hat, tol=self.tol))
                        kl_div = torch.mean(kl_divergence(latent_mu, latent_logsigma))

                    loss = kl_div + lambd * constraint
                    test_hist['loss'][-1] += loss.data.cpu().numpy()[0] / len(self.test_loader)
                    test_hist['reconstr'][-1] += constraint.data.cpu().numpy() / len(self.test_loader)
                    test_hist['KL'][-1] += kl_div.data.cpu().numpy() / len(self.test_loader)

            # update lr
            if self.scheduler is not None:
                self.scheduler.step(test_hist['loss'][-1])
            # stop
            if self.optimizer.param_groups[0]['lr'] <= 1e-6:
                break

            # visualization of training
            if self.verbose:
                display.clear_output(wait=True)
                draw_hist(train_hist, test_hist, lambd_hist)

            print("Epoch {} of {} took {:.3f}s".format(epoch + 1, self.num_epochs, time.time() - start_time))
            print("  training loss (in-iteration): \t{:.6f}".format(train_hist['loss'][-1]))
            print("  validation loss (in-iteration): \t{:.6f}".format(test_hist['loss'][-1]))

        return lambd_hist, train_hist, test_hist


class Train_Beta:
    def __init__(self, model, optimizer, scheduler, train_loader, test_loader, constraint_f=log_likelihood,
                 num_epochs=20, beta=torch.FloatTensor([1]), verbose=True, device='cpu', tol=1):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.constraint_f = constraint_f
        self.num_epochs = num_epochs
        self.beta = beta
        self.verbose = verbose
        self.device = device
        self.tol = tol

    def train(self):
        parametrized = self.model.parametrized
        self.model.to(self.device)
        train_hist = {'loss': [], 'reconstr': [], 'KL': []}
        test_hist = {'loss': [], 'reconstr': [], 'KL': []}

        beta = self.beta.to(self.device)
        for epoch in range(self.num_epochs):
            start_time = time.time()

            self.model.train(True)
            train_hist['loss'].append(0)
            train_hist['reconstr'].append(0)
            train_hist['KL'].append(0)

            for X_batch in self.train_loader:

                if isinstance(self.model, models.ConvolutionalDRAW):
                    X_batch = X_batch.to(self.device)
                    x_hat, kl, latent_mu, q_std = self.model(X_batch)
                    latent_logsigma = torch.pow(q_std, 2)
                    kl_div = torch.mean(kl.sum(dim=(1, 2, 3)))
                else:
                    X_batch = X_batch.reshape(self.train_loader.batch_size, -1).to(self.device)
                    if parametrized:
                        reconstruction_mu, reconstruction_logsigma, latent_mu, latent_logsigma, _ = \
                            self.model(X_batch)
                        constraint = - torch.mean(
                            self.constraint_f(X_batch, reconstruction_mu, reconstruction_logsigma))
                    else:
                        x_hat, latent_mu, latent_logsigma, _ = self.model(X_batch)
                        constraint = torch.mean(self.constraint_f(X_batch, x_hat, tol=self.tol))
                    kl_div = torch.mean(kl_divergence(latent_mu, latent_logsigma))
                loss = kl_div * beta + constraint

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                train_hist['loss'][-1] += loss.data.cpu().numpy()[0] / len(self.train_loader)
                train_hist['reconstr'][-1] += constraint.data.cpu().numpy() / len(self.train_loader)
                test_hist['KL'][-1] += kl_div.data.cpu().numpy() / len(self.test_loader)
            self.model.train(False)
            test_hist['loss'].append(0)
            test_hist['reconstr'].append(0)
            test_hist['KL'].append(0)
            with torch.no_grad():
                for X_batch in self.test_loader:

                    if isinstance(self.model, models.ConvolutionalDRAW):
                        X_batch = X_batch.to(self.device)
                        x_hat, kl = self.model(X_batch)
                        kl_div = torch.mean(kl.sum(dim=(1, 2, 3)))
                    else:
                        X_batch = X_batch.reshape(self.train_loader.batch_size, -1).to(self.device)
                        if parametrized:
                            reconstruction_mu, reconstruction_logsigma, latent_mu, latent_logsigma, _ = \
                                self.model(X_batch)
                            constraint = - torch.mean(
                                self.constraint_f(X_batch, reconstruction_mu, reconstruction_logsigma))
                        else:
                            x_hat, latent_mu, latent_logsigma, _ = self.model(X_batch)
                            constraint = torch.mean(self.constraint_f(X_batch, x_hat, tol=self.tol))
                        kl_div = torch.mean(kl_divergence(latent_mu, latent_logsigma))
                        loss = kl_div * beta + constraint
                    test_hist['loss'][-1] += loss.data.cpu().numpy()[0] / len(self.test_loader)
                    test_hist['reconstr'][-1] += constraint.data.cpu().numpy() / len(self.test_loader)
                    test_hist['KL'][-1] += kl_div.data.cpu().numpy() / len(self.test_loader)

            # update lr
            if self.scheduler is not None:
                self.scheduler.step(test_hist['loss'][-1])
            # stop
            if self.optimizer.param_groups[0]['lr'] <= 1e-6:
                break

            # visualization of training
            if self.verbose:
                display.clear_output(wait=True)
                draw_hist(train_hist, test_hist)

            print("Epoch {} of {} took {:.3f}s".format(epoch + 1, self.num_epochs, time.time() - start_time))
            print("  training loss (in-iteration): \t{:.6f}".format(train_hist['loss'][-1]))
            print("  validation loss (in-iteration): \t{:.6f}".format(test_hist['loss'][-1]))

        return train_hist, test_hist
