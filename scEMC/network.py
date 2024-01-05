from sklearn.cluster import KMeans
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from layers import NBLoss, ZINBLoss, MeanAct, DispAct
from utils import *
import math, os

def buildNetwork2(layers, type, activation="relu"):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        net.append(nn.BatchNorm1d(layers[i], affine=True))
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="selu":
            net.append(nn.SELU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
        elif activation=="elu":
            net.append(nn.ELU())
    return nn.Sequential(*net)

class scEMC(nn.Module):
    def __init__(self, input_dim1, input_dim2,
            encodeLayer1=[],
                 encodeLayer2=[],
                 decodeLayer1=[], decodeLayer2=[], tau=1., t=10, device="cuda",
            activation="elu", sigma1=2.5, sigma2=.1, alpha=1., gamma=1., phi1=0.0001, phi2=0.0001, cutoff = 0.5):
        super(scEMC, self).__init__()
        self.tau=tau
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.cutoff = cutoff
        self.activation = activation
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.alpha = alpha
        self.gamma = gamma
        self.phi1 = phi1
        self.phi2 = phi2
        self.t = t
        self.device = device
        self.encoder1 = buildNetwork2([input_dim1]+encodeLayer1, type="encode", activation=activation)
        self.encoder2 = buildNetwork2([input_dim2]+encodeLayer2, type="encode", activation=activation)
        self.decoder1 = buildNetwork2(decodeLayer1, type="decode", activation=activation)
        self.decoder2 = buildNetwork2(decodeLayer2, type="decode", activation=activation)
        self.dec_mean1 = nn.Sequential(nn.Linear(decodeLayer1[-1], input_dim1), MeanAct())
        self.dec_disp1 = nn.Sequential(nn.Linear(decodeLayer1[-1], input_dim1), DispAct())
        self.dec_mean2 = nn.Sequential(nn.Linear(decodeLayer2[-1], input_dim2), MeanAct())
        self.dec_disp2 = nn.Sequential(nn.Linear(decodeLayer2[-1], input_dim2), DispAct())
        self.dec_pi1 = nn.Sequential(nn.Linear(decodeLayer1[-1], input_dim1), nn.Sigmoid())
        self.dec_pi2 = nn.Sequential(nn.Linear(decodeLayer2[-1], input_dim2), nn.Sigmoid())
        self.zinb_loss = ZINBLoss()
        self.z_dim = decodeLayer1[0]
        self.trans_enc = nn.TransformerEncoderLayer(d_model= 2*encodeLayer1[-1], nhead=1, dim_feedforward=256)
        self.extract_layers = nn.TransformerEncoder(self.trans_enc, num_layers=1)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)
        
    def cal_latent(self, z):
        sum_y = torch.sum(torch.square(z), dim=1)
        num = -2.0 * torch.matmul(z, z.t()) + torch.reshape(sum_y, [-1, 1]) + sum_y
        num = num / self.alpha
        num = torch.pow(1.0 + num, -(self.alpha + 1.0) / 2.0)
        zerodiag_num = num - torch.diag(torch.diag(num))
        latent_p = (zerodiag_num.t() / torch.sum(zerodiag_num, dim=1)).t()
        return num, latent_p

    def kmeans_loss(self, z):
        tau = torch.tensor(self.tau).to('cuda:0')
        mu = torch.tensor(self.mu).to('cuda:0')
        dist1 = tau*torch.sum(torch.square(z.unsqueeze(1) - mu), dim=2)
        temp_dist1 = dist1 - torch.reshape(torch.mean(dist1, dim=1), [-1, 1])
        q = torch.exp(-temp_dist1)
        q = (q.t() / torch.sum(q, dim=1)).t()
        q = torch.pow(q, 2)
        q = (q.t() / torch.sum(q, dim=1)).t()
        dist2 = dist1 * q
        return dist1, torch.mean(torch.sum(dist2, dim=1))

    def target_distribution(self, q):
        p = q**2 / q.sum(0)
        return (p.t() / p.sum(1)).t()

    def forward(self, x1, x2):
        x1_1 = x1+torch.randn_like(x1)*self.sigma1
        x2_1 = x2+torch.randn_like(x2)*self.sigma2
        xh1 = self.encoder1(x1_1)
        xh2 = self.encoder2(x2_1)
        # h = torch.cat([xh1, xh2], dim=-1)
        h = self.extract_layers(torch.cat((xh1, xh2), 1))
        h = torch.cat([xh1, h], dim=-1)

        h1 = self.decoder1(h)
        mean1 = self.dec_mean1(h1)
        disp1 = self.dec_disp1(h1)
        pi1 = self.dec_pi1(h1)

        h2 = self.decoder2(h)
        mean2 = self.dec_mean2(h2)
        disp2 = self.dec_disp2(h2)
        pi2 = self.dec_pi2(h2)

        x1_2 = self.encoder1(x1)
        x2_2 = self.encoder2(x2)
        # h00 = torch.cat([x1_2, x2_2], dim=-1)
        h00 = self.extract_layers(torch.cat((x1_2, x2_2), 1))
        h00 = torch.cat([x1_2, h00], dim=-1)

        num, lq = self.cal_latent(h00)
        return h00, num, lq, mean1, mean2, disp1, disp2, pi1, pi2

        
    def encodeBatch(self, X1, X2, batch_size=256):
        encoded = []
        self.eval()
        num = X1.shape[0]
        num_batch = int(math.ceil(1.0*X1.shape[0]/batch_size))
        for batch_idx in range(num_batch):
            x1batch = X1[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
            x2batch = X2[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
            inputs1 = Variable(x1batch)
            inputs2 = Variable(x2batch)
            z,_,_,_,_,_,_,_,_ = self.forward(inputs1, inputs2)
            encoded.append(z.data)

        encoded = torch.cat(encoded, dim=0)
        return encoded

    def kldloss(self, p, q):
        c1 = -torch.sum(p * torch.log(q), dim=-1)
        c2 = -torch.sum(p * torch.log(p), dim=-1)
        return torch.mean(c1 - c2)

    def pretrain_autoencoder(self, X1, X_raw1, sf1, X2, X_raw2, sf2, 
            batch_size=256, lr=0.001, epochs=400, ae_save=True, ae_weights='AE_weights.pth.tar'):
        num_batch = int(math.ceil(1.0*X1.shape[0]/batch_size))
        dataset = TensorDataset(torch.Tensor(X1), torch.Tensor(X_raw1), torch.Tensor(sf1), torch.Tensor(X2), torch.Tensor(X_raw2), torch.Tensor(sf2))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print("Pretraining stage")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)
        num = X1.shape[0]
        for epoch in range(epochs):
            loss_val = 0
            recon_loss1_val = 0
            recon_loss2_val = 0
            kl_loss_val = 0
            for batch_idx, (x1_batch, x_raw1_batch, sf1_batch, x2_batch, x_raw2_batch, sf2_batch) in enumerate(dataloader):
                x1_tensor = Variable(x1_batch).to(self.device)
                x_raw1_tensor = Variable(x_raw1_batch).to(self.device)
                sf1_tensor = Variable(sf1_batch).to(self.device)
                x2_tensor = Variable(x2_batch).to(self.device)
                x_raw2_tensor = Variable(x_raw2_batch).to(self.device)
                sf2_tensor = Variable(sf2_batch).to(self.device)
                zbatch, z_num, lqbatch, mean1_tensor, mean2_tensor, disp1_tensor, disp2_tensor, pi1_tensor, pi2_tensor = self.forward(x1_tensor, x2_tensor)
                recon_loss1 = self.zinb_loss(x=x_raw1_tensor, mean=mean1_tensor, disp=disp1_tensor, pi=pi1_tensor, scale_factor=sf1_tensor)
                recon_loss2 = self.zinb_loss(x=x_raw2_tensor, mean=mean2_tensor, disp=disp2_tensor, pi=pi2_tensor, scale_factor=sf2_tensor)
                lpbatch = self.target_distribution(lqbatch)
                lqbatch = lqbatch + torch.diag(torch.diag(z_num))
                lpbatch = lpbatch + torch.diag(torch.diag(z_num))
                kl_loss = self.kldloss(lpbatch, lqbatch) 
                if epoch+1 >= epochs * self.cutoff:
                   loss = recon_loss1 + recon_loss2 + kl_loss * self.phi1
                else:
                   loss = recon_loss1 + recon_loss2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val += loss.item() * len(x1_batch)
                recon_loss1_val += recon_loss1.item() * len(x1_batch)
                recon_loss2_val += recon_loss2.item() * len(x2_batch)
                if epoch+1 >= epochs * self.cutoff:
                    kl_loss_val += kl_loss.item() * len(x1_batch)

            loss_val = loss_val/num
            recon_loss1_val = recon_loss1_val/num
            recon_loss2_val = recon_loss2_val/num
            kl_loss_val = kl_loss_val/num
            if epoch%self.t == 0:
               print('Pretrain epoch {}, Total loss:{:.6f}, ZINB loss1:{:.6f}, ZINB loss2:{:.6f}, KL loss:{:.6f}'.format(epoch+1, loss_val, recon_loss1_val, recon_loss2_val, kl_loss_val))

        if ae_save:
            torch.save({'ae_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, ae_weights)

    def save_checkpoint(self, state, index, filename):
        newfilename = os.path.join(filename, 'FTcheckpoint_%d.pth.tar' % index)
        torch.save(state, newfilename)

    def fit(self, X1, X_raw1, sf1, X2, X_raw2, sf2,lam1,lam2, y=None, lr=0.001, n_clusters = 16,
            batch_size=256, num_epochs=10, update_interval=1, tol=1e-3, save_dir=""):
        '''X: tensor data'''
        print("Clustering stage")
        X1 = torch.tensor(X1).to(self.device)
        X_raw1 = torch.tensor(X_raw1).to(self.device)
        sf1 = torch.tensor(sf1).to(self.device)
        X2 = torch.tensor(X2).to(self.device)
        X_raw2 = torch.tensor(X_raw2).to(self.device)
        sf2 = torch.tensor(sf2).to(self.device)
        self.mu = Parameter(torch.Tensor(n_clusters, self.z_dim), requires_grad=True)
        optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, rho=.95)
             
        print("Initializing cluster centers with kmeans.")
        kmeans = KMeans(n_clusters, n_init=20)
        Zdata = self.encodeBatch(X1, X2, batch_size=batch_size)
        #latent
        self.y_pred = kmeans.fit_predict(Zdata.data.cpu().numpy())
        self.y_pred_last = self.y_pred
        self.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))

        if y is not None:
            nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 4)
            ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 4)
            print('Initializing k-means: ARI= %.4f, NMI= %.4f' % (ari, nmi))
        
        self.train()
        num = X1.shape[0]
        num_batch = int(math.ceil(1.0*X1.shape[0]/batch_size))

        final_ami, final_nmi, final_ari, final_epoch = 0, 0, 0, 0
        
        stop = 0

        Loss_save = []
        for epoch in range(num_epochs):
            if epoch%update_interval == 0:
                Zdata = self.encodeBatch(X1, X2, batch_size=batch_size)
                dist, _ = self.kmeans_loss(Zdata)
                self.y_pred = torch.argmin(dist, dim=1).data.cpu().numpy()
                if y is not None:
                    #acc2 = np.round(cluster_acc(y, self.y_pred), 5)
                    final_nmi = nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 4)
                    final_ari = ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 4)
                    acc = np.round(cluster_acc(y, self.y_pred), 4)
                    pur = np.round(Purity_score(y, self.y_pred), 4)
                    final_epoch = epoch+1

                    print('Clustering   %d: ARi= %.4f, NMI= %.4f' % (epoch+1, ari, nmi))

                # check stop criterion
                delta_label = np.sum(self.y_pred != self.y_pred_last).astype(np.float32) / num
                self.y_pred_last = self.y_pred

                if epoch>0 and delta_label < tol:
                   #stop +=1
                   print('delta_label ', delta_label, '< tol ', tol)
                   #print("Stop + 1")
                   print("Reach tolerance threshold. Stopping training.")
                   break

            # train 1 epoch for clustering loss
            loss_val = 0.0
            recon_loss1_val = 0.0
            recon_loss2_val = 0.0
            cluster_loss_val = 0.0
            kl_loss_val = 0.0
            for batch_idx in range(num_batch):
                x1_batch = X1[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                x_raw1_batch = X_raw1[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                sf1_batch = sf1[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                x2_batch = X2[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                x_raw2_batch = X_raw2[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                sf2_batch = sf2[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]

                inputs1 = Variable(x1_batch)
                rawinputs1 = Variable(x_raw1_batch)
                sfinputs1 = Variable(sf1_batch)
                inputs2 = Variable(x2_batch)
                rawinputs2 = Variable(x_raw2_batch)
                sfinputs2 = Variable(sf2_batch)

                zbatch, z_num, lqbatch, mean1_tensor, mean2_tensor, disp1_tensor, disp2_tensor, pi1_tensor, pi2_tensor = self.forward(inputs1, inputs2)
                _, cluster_loss = self.kmeans_loss(zbatch)
                recon_loss1 = self.zinb_loss(x=rawinputs1, mean=mean1_tensor, disp=disp1_tensor, pi=pi1_tensor, scale_factor=sfinputs1)
                recon_loss2 = self.zinb_loss(x=rawinputs2, mean=mean2_tensor, disp=disp2_tensor, pi=pi2_tensor, scale_factor=sfinputs2)
                target2 = self.target_distribution(lqbatch)
                lqbatch = lqbatch + torch.diag(torch.diag(z_num))
                target2 = target2 + torch.diag(torch.diag(z_num))
                kl_loss = self.kldloss(target2, lqbatch)
                lc = kl_loss * self.phi2 + cluster_loss * self.gamma
                lr = recon_loss1 + recon_loss2
                loss = lam1* lc+lam2 *lr

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                cluster_loss_val += cluster_loss.data * len(inputs1)
                recon_loss1_val += recon_loss1.data * len(inputs1)
                recon_loss2_val += recon_loss2.data * len(inputs2)
                kl_loss_val += kl_loss.data * len(inputs1)
                loss_val += loss.data * len(inputs1)


            if epoch%self.t == 0:
               print("#Epoch %d: Total: %.6f Clustering Loss: %.6f ZINB Loss1: %.6f ZINB Loss2: %.6f KL Loss: %.6f" % (
                     epoch + 1, loss_val / num, cluster_loss_val / num, recon_loss1_val / num, recon_loss2_val / num, kl_loss_val / num))

            Loss_save.append(loss.item())


        return self.y_pred,  final_nmi, final_ari, final_epoch
