import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class Generator_dis(nn.Module):
    def __init__(self, DIM=128, z_dim=16, img_shape=(324, 324), final_shape = [9, 9], cl_tensor=None):
        super(Generator_dis, self).__init__()
        self.DIM = DIM
        self.final_shape = final_shape
        self.final_dim = np.prod(self.final_shape)
        self.img_shape = img_shape
        preprocess = nn.Sequential(
            nn.Conv2d(z_dim, 4 * DIM, 1, 1),
            nn.BatchNorm2d(4 * DIM),
            nn.ReLU(True),
        )

        block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * DIM, 2 * DIM, 4, stride=2, padding=3),
            nn.BatchNorm2d(2 * DIM),
            nn.ReLU(True),
            nn.ConvTranspose2d(2 * DIM, 2 * DIM, 4, stride=2, padding=3),
            nn.BatchNorm2d(2 * DIM),
            nn.ReLU(True),
            nn.ConvTranspose2d(2 * DIM, 2 * DIM, 4, stride=2, padding=3),
            nn.BatchNorm2d(2 * DIM),
            nn.ReLU(True),
            nn.ConvTranspose2d(2 * DIM, 2 * DIM, 4, stride=2, padding=3),
            nn.BatchNorm2d(2 * DIM),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * DIM, DIM, 4, stride=2, padding=3),
            nn.BatchNorm2d(DIM),
            nn.ReLU(True),
        )
        # deconv_out = nn.ConvTranspose2d(DIM, self.cl_num, 4, stride=2, padding=3)
        deconv_out = nn.ConvTranspose2d(DIM, 3, 4, stride=2, padding=3)

        self.preprocess = preprocess
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.tanh = nn.Tanh()

    def forward(self, input):
        # print(input.shape)
        output = self.preprocess(input)
        # print(output.shape)
        output = self.block1(output)
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.tanh(output)
        # output = F.sigmoid(output)
        return output


class Discriminator_dis(nn.Module):
    def __init__(self, DIM=128, img_shape=(324, 324), final_shape= [9, 9], cl_tensor=None):
        super(Discriminator_dis, self).__init__()
        self.DIM = DIM
        # self.final_shape = (np.array(img_shape) / 32).astype(np.int64)
        self.final_shape = final_shape
        self.final_dim = np.prod(self.final_shape)
        # self.img_shape  = self.final_shape * 32
        self.img_shape = img_shape
        self.main = nn.Sequential(
            # nn.Conv2d(self.cl_num, DIM, 3, 2, padding=3),
            nn.Conv2d(3, DIM, 4, 2, padding=3),
            nn.LeakyReLU(),
            nn.Conv2d(DIM, 2 * DIM, 4, 2, padding=3),
            nn.LeakyReLU(),
            nn.Conv2d(2 * DIM, 2 * DIM, 4, 2, padding=3),
            nn.LeakyReLU(),
            nn.Conv2d(2 * DIM, 2 * DIM, 4, 2, padding=3),
            nn.LeakyReLU(),
            nn.Conv2d(2 * DIM, 2 * DIM, 4, 2, padding=3),
            nn.LeakyReLU(),
            nn.Conv2d(2 * DIM, 4 * DIM, 4, 2, padding=3),
            nn.LeakyReLU(),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(5 * DIM, DIM, 1, 1),
            nn.LeakyReLU(),
        )

        self.linear = nn.Sequential(nn.Linear(self.final_dim * DIM, 512),
                                    nn.LeakyReLU(),
                                    nn.Linear(512, 512),
                                    nn.LeakyReLU(),
                                    nn.Linear(512, 1),
                                    )

    def forward(self, input, z):
        output = self.main(input)
        output = torch.cat([output, z], dim=1)
        output = self.block2(output)
        output = output.view(output.shape[0], self.final_dim * self.DIM)
        output = self.linear(output)
        return output



class GAN_dis(nn.Module):
    def __init__(self, DIM=128, z_dim=16, img_shape=(324, 324), final_shape=(9,9), cl_tensor=None, args=None):
        super(GAN_dis, self).__init__()
        self.DIM = DIM
        self.z_dim = z_dim
        self.final_shape = final_shape
        self.final_dim = np.prod(self.final_shape)
        self.img_shape = img_shape
        self.G = Generator_dis(self.DIM, self.z_dim, self.img_shape,final_shape=final_shape)
        self.D = Discriminator_dis(self.DIM, self.img_shape,final_shape=final_shape)

    def forward(self, input):
        pass

    def get_loss(self, y, z, gp=0.0):
        if z.shape[0] > 1:
            z_prime = torch.cat((z[1:], z[:1]), dim=0)
        else:
            z_prime = z.new(z.shape).normal_()
        pj = self.D(y, z)
        pm = self.D(y, z_prime)
        Ej = -F.softplus(-pj).mean()
        Em = F.softplus(pm).mean()
        loss = (Em - Ej)
        if gp > 0:
            return loss + gp * calc_gradient_penalty(self.D, y), pj, pm
        else:
            return loss, pj, pm
    
    # l1
    def get_loss_ctr_3(self, model, image_batch):
        out = model(image_batch)
        batch_size =  image_batch.shape[0]//4
        fc_features = model.get_buffer('fc_features')
        loss = F.triplet_margin_loss(fc_features[batch_size:2*batch_size], fc_features[2*batch_size:3*batch_size],
                                    fc_features[-batch_size:],p=1, margin=10000, reduction='mean')    
        loss -= F.l1_loss(fc_features[2*batch_size:3*batch_size], fc_features[:batch_size]) 
        return loss     
        
    # cos sim
    def get_loss_ctr_2(self, model, image_batch):
        out = model(image_batch)
        batch_size =  image_batch.shape[0]//4
        fc_features = model.get_buffer('fc_features')
        loss = F.triplet_margin_with_distance_loss(fc_features[batch_size:2*batch_size], 
                                    fc_features[-batch_size:], fc_features[2*batch_size:3*batch_size], 
                                    distance_function=F.cosine_similarity,margin=10000, reduction='mean')    
        loss += F.cosine_similarity(fc_features[2*batch_size:3*batch_size], fc_features[:batch_size]).mean() 
        return loss
    
    # l2 
    def get_loss_ctr_1(self, model, image_batch, key = None):
        out = model(image_batch)
        batch_size =  image_batch.shape[0]//4
        fc_features = eval(f"model.net.{key}").get_buffer('fc_features')
        loss = F.mse_loss(fc_features[batch_size:2*batch_size], fc_features[2*batch_size:3*batch_size] )    
        loss += -F.mse_loss(fc_features[batch_size:2*batch_size], fc_features[-batch_size:])    
        # loss = F.triplet_margin_loss(fc_features[batch_size:2*batch_size], fc_features[2*batch_size:3*batch_size],
        #                             fc_features[-batch_size:], margin=10000, reduction='mean')    
        # loss += - F.mse_loss(fc_features[2*batch_size:3*batch_size], fc_features[:batch_size]) 
        return loss
    
    def get_loss_ctr(self, model, image_batch, lab_batch):
        # pp = self.D_ctr(patch )
        # pd = self.D_ctr(diff )

        out = model(image_batch)
        batch_size = image_batch.shape[0] // 3
        fc_features = model.get_buffer('fc_features')
        loss = F.triplet_margin_loss(fc_features[:batch_size], fc_features[-batch_size:],
                                     fc_features[batch_size:-batch_size], margin=6, reduction='mean')

        # out = model(image_batch)
        loss += F.cross_entropy(out[:batch_size],lab_batch[:batch_size])

        # loss =  F.cosine_similarity(pp,pd,1).abs().mean()
        # loss = F.cross_entropy(pp,pp.new_ones(pp.shape[0],dtype=torch.long)) \
        #     + F.cross_entropy(pd,pd.new_zeros(pd.shape[0],dtype=torch.long))
        return loss

    def generate(self, z=None, batch_size=None, sample_mode=None):
        # z = self.z
        if z is None:
            z = torch.randn(batch_size, self.DIM,
                            self.final_shape[0], self.final_shape[1])
            z = z.to(self.D.linear.weight)

        x = self.G(z)
        x_proj = x
        return x_proj

    def discriminate(self, x):
        return self.D(x)

    def anneal(self, steps=1):
        self.temp = max(self.temp * np.exp(- steps *
                        self.anneal_rate), self.temp_min)


def calc_gradient_penalty(net, real_data):

    interpolates = real_data.clone().detach().requires_grad_()
    disc_interpolates = net(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(
                                        disc_interpolates.size(), device=disc_interpolates.device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
