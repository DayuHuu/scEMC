from time import time
import argparse
import load_data as loader
from network import scEMC
from preprocess import read_dataset, normalize
from utils import *

if __name__ == "__main__":
    my_data_dic = loader.ALL_data
    for i_d in my_data_dic:
        data_para = my_data_dic[i_d]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        seed = seed_set.seed  # 改
        parser = argparse.ArgumentParser(description='scEMC')
        parser.add_argument('--n_clusters', default=data_para['K'], type=int)
        parser.add_argument('--lr', default=1, type=float) #1
        parser.add_argument('-el1', '--encodeLayer1', nargs='+', default=[256, 64, 32, 8])
        parser.add_argument('-el2', '--encodeLayer2', nargs='+', default=[256, 64, 32, 8])
        parser.add_argument('-dl1', '--decodeLayer1', nargs='+', default=[24, 64, 256])
        parser.add_argument('-dl2', '--decodeLayer2', nargs='+', default=[24, 20])
        parser.add_argument('--dataset', default=data_para)
        parser.add_argument("--view_dims", default=data_para['n_input'])
        parser.add_argument('--name', type=str, default=data_para[1])
        parser.add_argument('--cutoff', default=0.5, type=float,
                            help='Start to train combined layer after what ratio of epoch')
        parser.add_argument('--batch_size', default=256, type=int)
        parser.add_argument('--maxiter', default=500, type=int)
        parser.add_argument('--pretrain_epochs', default=400, type=int)  # 400
        #聚类损失
        parser.add_argument('--gamma', default=.1, type=float,
                            help='coefficient of clustering loss') #.1
        parser.add_argument('--tau', default=1., type=float,
                            help='fuzziness of clustering loss')
        parser.add_argument('--phi1', default=0.001, type=float,
                            help='预训练coefficient of KL loss')
        #KL损失
        parser.add_argument('--phi2', default=0.001, type=float,
                            help='coefficient of KL loss') #0.001
        parser.add_argument('--update_interval', default=1, type=int)
        parser.add_argument('--tol', default=0.001, type=float)
        parser.add_argument('--ae_weights', default=None)
        parser.add_argument('--save_dir', default='results/')
        parser.add_argument('--ae_weight_file', default='AE_weights_1.pth.tar')
        parser.add_argument('--resolution', default=0.2, type=float)
        parser.add_argument('--n_neighbors', default=30, type=int)
        parser.add_argument('--embedding_file', action='store_true', default=True)
        parser.add_argument('--prediction_file', action='store_true', default=False)
        parser.add_argument('--sigma1', default=2.5, type=float)
        parser.add_argument('--sigma2', default=1.5, type=float)
        parser.add_argument('--f1', default=2000, type=float, help='Number of mRNA after feature selection')
        parser.add_argument('--f2', default=2000, type=float, help='Number of ADT/ATAC after feature selection')
        parser.add_argument('--filter1', action='store_true', default=False, help='Do mRNA selection')
        parser.add_argument('--filter2', action='store_true', default=False, help='Do ADT/ATAC selection')
        parser.add_argument('--run', default=1, type=int)
        parser.add_argument('--device', default='cuda')
        parser.add_argument('--lam1', default=1, type=float)
        parser.add_argument('--lam2', default=1, type=float)


        args = parser.parse_args()
        X, Y = loader.load_data(args.dataset)
        labels = Y[0].copy().astype(np.int32)
        x1 = np.array(X[0])
        x2 = np.array(X[1])
        y = labels
        y = np.reshape(y, -1)
        print(args)

        # Gene filter
        if args.filter1:
            importantGenes = geneSelection(x1, n=args.f1, plot=False)
            x1 = x1[:, importantGenes]
        if args.filter2:
            importantGenes = geneSelection(x2, n=args.f2, plot=False)
            x2 = x2[:, importantGenes]

        # preprocessing scRNA-seq read counts matrix
        adata1 = sc.AnnData(x1)
        adata1.obs['Group'] = y

        adata1 = read_dataset(adata1,
                              transpose=False,
                              test_split=False,
                              copy=True)

        adata1 = normalize(adata1,
                           size_factors=True,
                           normalize_input=True,
                           logtrans_input=True)
        adata2 = sc.AnnData(x2)
        adata2.obs['Group'] = y
        adata2 = read_dataset(adata2,
                              transpose=False,
                              test_split=False,
                              copy=True)
        adata2 = normalize(adata2,
                           filter_min_counts=None,
                           size_factors=False,
                           normalize_input=False,
                           logtrans_input=False)

        input_size1 = adata1.n_vars
        input_size2 = adata2.n_vars


        print(args)

        encodeLayer1 = list(map(int, args.encodeLayer1))
        encodeLayer2 = list(map(int, args.encodeLayer2))
        decodeLayer1 = list(map(int, args.decodeLayer1))
        decodeLayer2 = list(map(int, args.decodeLayer2))

        model = scEMC(input_dim1=input_size1, input_dim2=input_size2, tau=args.tau,
                               encodeLayer1=encodeLayer1,
                               encodeLayer2=encodeLayer2,
                               decodeLayer1=decodeLayer1, decodeLayer2=decodeLayer2,
                               activation='elu', sigma1=args.sigma1, sigma2=args.sigma2, gamma=args.gamma,
                               cutoff=args.cutoff, phi1=args.phi1, phi2=args.phi2, device=args.device).to(args.device)


        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        t0 = time()
        if args.ae_weights is None:
            model.pretrain_autoencoder(X1=adata1.X, X_raw1=adata1.raw.X, sf1=adata1.obs.size_factors,
                                       X2=adata2.X, X_raw2=adata2.raw.X, sf2=adata2.obs.size_factors,
                                       batch_size=args.batch_size,
                                       epochs=args.pretrain_epochs, ae_weights=args.ae_weight_file)
        else:
            if os.path.isfile(args.ae_weights):
                print("==> loading checkpoint '{}'".format(args.ae_weights))
                checkpoint = torch.load(args.ae_weights)
                model.load_state_dict(checkpoint['ae_state_dict'])
            else:
                print("==> no checkpoint found at '{}'".format(args.ae_weights))
                raise ValueError

        print('Pretraining time: %d seconds.' % int(time() - t0))

        # get k
        latent = model.encodeBatch(torch.tensor(adata1.X).to(args.device), torch.tensor(adata2.X).to(args.device))
        latent = latent.cpu().numpy()
        if args.n_clusters == -1:
            n_clusters = GetCluster(latent, res=args.resolution, n=args.n_neighbors)
        else:
            print("n_cluster is defined as " + str(args.n_clusters))
            n_clusters = args.n_clusters

        y_pred,  _, _, _ = model.fit(X1=adata1.X, X_raw1=adata1.raw.X, sf1=adata1.obs.size_factors,
                                       X2=adata2.X, X_raw2=adata2.raw.X, sf2=adata2.obs.size_factors, y=y,
                                       n_clusters=n_clusters, batch_size=args.batch_size, num_epochs=args.maxiter,
                                       update_interval=args.update_interval, tol=args.tol, lr=args.lr,
                                       save_dir=args.save_dir,lam1=args.lam1,lam2=args.lam2)
        print('Total time: %d seconds.' % int(time() - t0))

        if args.prediction_file:
            y_pred_ = best_map(y, y_pred) - 1
            np.savetxt(args.save_dir + "/" + str(args.run) + "_pred.csv", y_pred_, delimiter=",")

        if args.embedding_file:
            final_latent = model.encodeBatch(torch.tensor(adata1.X).to(args.device),
                                             torch.tensor(adata2.X).to(args.device))
            final_latent = final_latent.cpu().numpy()
            np.savetxt(args.save_dir + "/" + str(args.run) + "_embedding.csv", final_latent, delimiter=",")

        y_pred_ = best_map(y, y_pred)
        nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 4)
        ari = np.round(metrics.adjusted_rand_score(y, y_pred), 4)
        acc = np.round(cluster_acc(y, y_pred), 4)
        pur = np.round(Purity_score(y, y_pred), 4)
        print('Final: ARI= %.4f, NMI= %.4f, ACC= %.4f, PUR= %.4f' % (ari, nmi, acc, pur))

        my_dic2 = dict({'View': 'multi', 'ARI': ari, 'NMI': nmi, 'ACC': acc, 'PUR': pur})  #

        f = open("./result/{}.txt".format(args.name), "a+")
        f.write(str(args))
        f.write("\n")
        f.write(str(seed))
        f.write("\n")
        f.write(str(my_dic2) + '\r')




