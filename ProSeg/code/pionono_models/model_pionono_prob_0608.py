import torch
from torch import nn
import numpy as np
from Probabilistic_Unet_Pytorch.utils import l2_regularisation
from pionono_models.model_headless import UnetHeadless
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LatentVariable(nn.Module):
    """
    This module defines the random latent variable z with distribution q(z|r) with r being the rater.
    """
    def __init__(self, num_annotators, latent_dims=8, prior_mu_value=0.0, prior_sigma_value=1.0, z_posterior_init_sigma=0.0):
        super(LatentVariable, self).__init__()
        #################Rearrange the distribution##################
        
        #prior_alpha = torch.ones(latent_dims)
        prior_alpha = torch.tensor([[10.0,0.1,0.1,0.1,10.0,0.1,0.1,0.1],[0.1,10.0,0.1,0.1,0.1,10.0,0.1,0.1],[0.1,0.1,10.0,0.1,0.1,0.1,10.0,0.1],[0.1,0.1,0.1,10.0, 0.1,0.1,0.1,10.0]])
        self.prior_alpha = torch.nn.Parameter(prior_alpha)
        self.prior_alpha.requires_grad = False
        
        posterior_alpha = torch.tensor([[1.0,0.1,0.1,0.1,1.0,0.1,0.1,0.1],[0.1,1.0,0.1,0.1,0.1,1.0,0.1,0.1],[0.1,0.1,1.0,0.1,0.1,0.1,1.0,0.1],[0.1,0.1,0.1,1.0, 0.1,0.1,0.1,1.0]])
        self.posterior_alpha = torch.nn.Parameter(posterior_alpha)
        self.posterior_alpha.requires_grad = True
        
        mu = prior_mu_value      # 原始均值
        sigma = prior_sigma_value   # 原始标准差

        # 子高斯分布参数
        K = num_annotators  # 子高斯分布数量
        range_factor = 2.6          # 覆盖范围倍数 (覆盖范围 = range_factor * sigma)
        delta = range_factor * sigma / (K - 1) # 子分布均值覆盖范围
        #delta_mu = 2 * delta / K  # 子分布均值间隔

        # 设置子分布的均值
        sub_means = [3+mu + delta * (i - (K + 1) / 2) for i in range(1, K + 1)]

        # 计算子分布的方差
        mean_offset_variance = np.mean([(m - mu)**2 for m in sub_means])
        sub_variance = sigma**2 - mean_offset_variance

        #if sub_variance <= 0:
        #    raise ValueError("Sub-distribution variance is non-positive. Reduce range_factor or adjust K.")

        sub_sigma = np.sqrt(np.abs(sub_variance))
        prior_mu_value = sub_means 
        prior_sigma_value = sub_sigma

        # 子分布的均值和方差
        #sub_distributions = []
        # prior_mu_value = []
        # prior_sigma_value = []
        # for i in range(K):
        #     sub_mu = mu + (i - (K + 1) / 2) * delta  # 子分布均值
        #     #sub_sigma = sigma / np.sqrt(K)             # 子分布方差
        #     #sub_distributions.append((sub_mu, sub_sigma))
        #     prior_mu_value.append(sub_mu)
        #     prior_sigma_value.append(sub_sigma)
        ##############################################################
    
        self.latent_dims = latent_dims
        self.no_annotators = num_annotators
        prior_mu, prior_cov = self._init_distributions(prior_mu=prior_mu_value, prior_sigma=prior_sigma_value)
        self.prior_mu = torch.nn.Parameter(prior_mu)
        self.prior_covtril = torch.nn.Parameter(prior_cov)
        self.prior_mu.requires_grad = False
        self.prior_covtril.requires_grad = False
        #############################################################
        post_mu_value = []
        post_sigma_value = []
        for i in range(K):
            post_mu_value.append(np.random.standard_normal(size=[1, latent_dims])*z_posterior_init_sigma + prior_mu_value[i])
        #post_mu_value = np.array(post_mu_value)
        post_mu_value = np.array(prior_mu_value)
        #post_sigma_value = prior_sigma_value
        post_sigma_value = prior_sigma_value/10
        ############################################################
        posterior_mu, posterior_cov = self._init_distributions(prior_mu=post_mu_value, prior_sigma=post_sigma_value)
        self.posterior_mu = torch.nn.Parameter(posterior_mu)
        self.posterior_covtril = torch.nn.Parameter(posterior_cov)
        self.posterior_mu.requires_grad = True
        self.posterior_covtril.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(latent_dims, 32),
            nn.ReLU(),
            nn.Linear(32, num_annotators),
            nn.Softmax(dim=1)
        )
        self.name = 'LatentVariable'

    def _init_distributions(self, prior_mu=0.0, prior_sigma=1.0):
        mu_list = []
        cov_list = []
        prior_mu = np.array(prior_mu)
        prior_sigma = np.array(prior_sigma)
        for a in range(self.no_annotators):
            if prior_mu.size > 1:
                mu = prior_mu[a]
            else:
                mu = prior_mu
            if prior_sigma.size > 1:
                sigma = prior_sigma[a]
            else:
                sigma = prior_sigma
            mu_a = np.ones(self.latent_dims)*mu
            # we use sigma values (instead of sigma+sigma) because we pass this matrix as tril matrix L
            # this makes the sigma value of the cov matrix squared
            cov_a = np.eye(self.latent_dims) * (sigma)
            mu_list.append(mu_a)
            cov_list.append(cov_a)
        mu_list = torch.tensor(np.array(mu_list))
        covtril_list = torch.tensor(np.array(cov_list))
        return mu_list, covtril_list

    def forward(self, annotator, sample=True):
        z = torch.zeros([len(annotator), self.latent_dims]).to(device)
        annotator = annotator.long()
        for i in range(len(annotator)):
            a = annotator[i]
            #dist_a = torch.distributions.multivariate_normal.MultivariateNormal(self.posterior_mu[a],
            #                                                                    scale_tril=torch.tril(self.posterior_covtril[a]))
            #dist_a = torch.distributions.Dirichlet(torch.nn.functional.softplus(self.posterior_alpha[a]))
            #if sample:
            #z_i = dist_a.rsample()
            #else:
            #    z_i = dist_a.loc
            #z[i] = z_i
            z[i] = self.posterior_alpha[a]
        self.latent_vectors = z
        return z
    
    def prior_forward(self, annotator, sample=True):
        z = torch.zeros([len(annotator), self.latent_dims]).to(device)
        annotator = annotator.long()
        for i in range(len(annotator)):
            a = annotator[i]
            # dist_a = torch.distributions.multivariate_normal.MultivariateNormal(self.prior_mu[a],
            #                                                                     scale_tril=torch.tril(self.prior_covtril[a]))
            dist_a = torch.distributions.Dirichlet(self.prior_alpha[a])
            if sample:
                z_i = dist_a.rsample()
            else:
                z_i = dist_a.loc
            z[i] = z_i
        return z
    
    def get_class_ce_loss(self, annotator):
        annotator = annotator.long()
        classes = self.classifier(self.latent_vectors)
        #annotator_one_hot = torch.nn.functional.one_hot(annotator, num_classes=self.no_annotators).float()
        #print(classes.shape, annotator.shape)
        #loss = nn.CrossEntropyLoss()(classes, annotator)
        #print(self.latent_vectors)
        loss = nn.CrossEntropyLoss()(classes, annotator)
        #print(loss)
        return loss

    def get_kl_loss(self, annotator):
        kl_loss = torch.zeros([len(annotator)]).to(device)
        annotator = annotator.long()
        for i in range(len(annotator)):
            a = annotator[i]
            #dist_a_posterior = torch.distributions.multivariate_normal.MultivariateNormal(self.posterior_mu[a],
            #                                                                              scale_tril=torch.tril(self.posterior_covtril[a]))
            #dist_a_prior = torch.distributions.multivariate_normal.MultivariateNormal(self.prior_mu[a],
            #                                                                          scale_tril=torch.tril(self.prior_covtril[a]))

            dist_a_posterior = torch.distributions.Dirichlet(torch.nn.functional.softplus(self.posterior_alpha[a]))
            dist_a_prior = torch.distributions.Dirichlet(self.prior_alpha[a])
            kl_loss[i] = torch.distributions.kl_divergence(dist_a_posterior, dist_a_prior)
        kl_mean = torch.mean(kl_loss)
        return kl_mean

class PiononoHead(nn.Module):
    """
    The Segmentation head combines the sample taken from the latent space,
    and feature map by concatenating them along their channel axis.
    """
    def __init__(self, num_filters_last_layer, latent_dim, num_output_channels, num_classes, no_convs_fcomb,
                 head_kernelsize, head_dilation, use_tile=True):
        super(PiononoHead, self).__init__()
        self.num_channels = num_output_channels #output channels
        self.num_classes = num_classes
        self.channel_axis = 1
        self.spatial_axes = [2,3]
        self.num_filters_last_layer = num_filters_last_layer
        self.latent_dim = latent_dim
        self.use_tile = use_tile
        self.no_convs_fcomb = no_convs_fcomb
        self.head_kernelsize = head_kernelsize
        self.name = 'PiononoHead'
        self.statistic_head = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            ) for _ in range(4)  # 创建三个相同的网络
        ])
        mu = 0      # 原始均值
        sigma = 8   # 原始标准差

        # 子高斯分布参数
        K = 4  # 子高斯分布数量
        range_factor = 2.6          # 覆盖范围倍数 (覆盖范围 = range_factor * sigma)
        delta = range_factor * sigma / (K - 1) # 子分布均值覆盖范围
        #delta_mu = 2 * delta / K  # 子分布均值间隔

        # 设置子分布的均值
        sub_means = [mu + delta * (i - (K + 1) / 2) for i in range(1, K + 1)]

        # 计算子分布的方差
        mean_offset_variance = np.mean([(m - mu)**2 for m in sub_means])
        sub_variance = sigma**2 - mean_offset_variance

        if sub_variance <= 0:
            raise ValueError("Sub-distribution variance is non-positive. Reduce range_factor or adjust K.")

        #sub_sigma = [np.sqrt(sub_variance)/10]*4
        sub_sigma = [1.0]*4
        sub_means = [0.0]*4
        
        self.posterior_mu = torch.nn.Parameter(torch.tensor(np.array(sub_means)))
        self.posterior_covtril = torch.nn.Parameter(torch.tensor(np.array(sub_sigma)))
        self.posterior_mu.requires_grad = False
        self.posterior_covtril.requires_grad = False

        if self.use_tile:
            layers = []

            # #Decoder of N x a 1x1 convolution followed by a ReLU activation function except for the last layer
            # layers.append(nn.Conv2d(int(16/2)+self.latent_dim, self.num_filters_last_layer,
            #                        kernel_size=self.head_kernelsize, dilation=head_dilation, padding='same'))
            
            layers.append(nn.Conv2d(self.num_filters_last_layer+self.latent_dim, self.num_filters_last_layer,
                                   kernel_size=self.head_kernelsize, dilation=head_dilation, padding='same'))
            # #Decoder of N x a 1x1 convolution followed by a ReLU activation function except for the last layer
            # layers.append(nn.Conv2d(self.num_filters_last_layer+1, self.num_filters_last_layer,
            #                        kernel_size=self.head_kernelsize, dilation=head_dilation, padding='same'))

            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_fcomb-2):
                layers.append(nn.Conv2d(self.num_filters_last_layer, self.num_filters_last_layer,
                                        kernel_size=self.head_kernelsize, dilation=head_dilation, padding='same'))
                layers.append(nn.ReLU(inplace=True))

            self.layers = nn.Sequential(*layers)
            self.last_layer = nn.Conv2d(self.num_filters_last_layer, self.num_classes, kernel_size=self.head_kernelsize,
                                        dilation=head_dilation, padding='same')
            self.activation = torch.nn.Softmax(dim=1)

            self.layers.apply(self.initialize_weights)
            self.last_layer.apply(self.initialize_weights)
        # self.Reconstruction_head = nn.Sequential(
        #     UnetHeadless(8).to(device),
        #     nn.Conv2d(16, 3, kernel_size=1)
        # )
        self.remap_head = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            ) for _ in range(4)  # 创建三个相同的网络
        ])
        
        self.reconstruction_head = nn.Sequential(
            UnetHeadless(8).to(device),
            nn.Conv2d(16, 3, kernel_size=1)
        )

    def initialize_weights(self, module):
        for m in module.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(
            np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(device)
        return torch.index_select(a, dim, order_index)

    def reparameterize(self, feature_map):
        # 8 is the number of channel
        mu = feature_map[:, :16, :, :]
        log_var = feature_map[:, 16:, :, :]
        #print(mu.shape, log_var.shape)
        #torch.Size([12, 4, 128, 128]) torch.Size([12, 12, 128, 128])
        std = torch.exp(0.5 * log_var)  # 计算标准差 σ
        epsilon = torch.randn_like(std)  # 采样 ε ~ N(0,1)
        z = mu + std * epsilon  # 重参数化
        return z
    
    def get_reconstruction_loss(self, x, annotator):
        """
        计算重构损失
        :param x: 输入数据
        :param x_recon: 重构数据
        :return: 重构损失
        """
        remap_feature = self.get_id_features_down(annotator,  self.latent_feature_map, head=self.remap_head)

        x_recon = self.reconstruction_head(remap_feature)
        #print(x.shape, x_recon.shape)
        #torch.Size([12, 16, 128, 128]) torch.Size([12, 16, 128, 128])
        batch_size = x.size(0)
        #print(x.shape, x_recon.shape)
        #torch.Size([12, 16, 128, 128]) torch.Size([12, 16, 128, 128])
        recon_loss = nn.MSELoss(reduction='mean')(x_recon, x) / batch_size
        return recon_loss
    
    
    def kl_divergence_gaussian(self, mu_q, log_var_q, mu_p=0.0, log_var_p=0.0):
        """
        计算两个高斯分布的 KL 散度 KL(q || p)
        :param mu_q: 变分分布的均值 (q)
        :param log_var_q: 变分分布的对数方差 (q)
        :param mu_p: 先验分布的均值 (p)，默认为 0
        :param log_var_p: 先验分布的对数方差 (p)，默认为 0（对应 sigma_p = 1）
        :return: KL 散度
        """
        sigma_q2 = torch.exp(log_var_q)  # 变分分布的方差 σ_q^2
        sigma_p2 = torch.exp(log_var_p)  # 先验分布的方差 σ_p^2

        kl = 0.5 * (log_var_p - log_var_q + (sigma_q2 + (mu_q - mu_p) ** 2) / sigma_p2 - 1)
        return kl.mean()  # 计算 KL 散度的总和
    
    def get_kl_loss(self, annotator, feature):
        #print(feature.shape)
        #torch.Size([12, 16, 128, 128])
        mu_q = feature[:, :16, :, :]
        log_var_q = feature[:, 16:, :, :]
        kl_loss = torch.zeros([len(annotator)]).to(device)
        annotator = annotator.long()
        #annotator_ids = self.map_annotators_to_correct_id(annotator_ids, annotator_list)
        #print(annotator)
        for i in range(len(annotator)):
            a = annotator[i]
            #print(a.item())
            mu_p = self.posterior_mu[int(a.item())]
            log_var_p = self.posterior_covtril[int(a.item())]
            kl_loss[i] = self.kl_divergence_gaussian(mu_q, log_var_q, mu_p, log_var_p)
        kl_mean = torch.mean(kl_loss)
        return kl_mean

    def get_id_features_expand(self, annotator, feature, head):
        bs, c, h, w = feature.shape
        feature_id = torch.zeros((bs, c*2, h, w)).to(device)
        annotator = annotator.long()
        for i in range(len(annotator)):
            a = annotator[i]
            feature_id[i] = head[a](feature[i].unsqueeze(0)).squeeze(0)
        return feature_id

    def get_id_features_down(self, annotator, feature, head):
        bs, c, h, w = feature.shape
        feature_id = torch.zeros((bs, int(c/2), h, w)).to(device)
        annotator = annotator.long()
        for i in range(len(annotator)):
            a = annotator[i]
            feature_id[i] = head[a](feature[i].unsqueeze(0)).squeeze(0)
        return feature_id

    def forward(self, feature_map, z, ids, use_softmax=True):
        #print(feature_map.shape, z.shape)
        """
        Z is batch_size x latent_dim and feature_map is batch_size x no_channels x H x W.
        So broadcast Z to batch_size x latent_dim x H x W. Behavior is exactly the same as tf.tile (verified)
        """
        #print(feature_map.shape, z.shape)
        feature_map = self.get_id_features_expand(ids, feature_map, head=self.statistic_head)
        #print("feature.shape, z.shape",feature_map.shape, z.shape)
        #print(feature_map.shape, z.shape)
        #feature.shape, z.shape torch.Size([12, 16, 128, 128]) torch.Size([12, 10])
        self.feature_map = feature_map
        feature_map = self.reparameterize(feature_map)
        self.latent_feature_map = feature_map
        #print(feature_map.shape)
        if self.use_tile:
            z = torch.unsqueeze(z, 2)
            z = self.tile(z, 2, feature_map.shape[self.spatial_axes[0]])
            z = torch.unsqueeze(z, 3)
            z = self.tile(z, 3, feature_map.shape[self.spatial_axes[1]])
            #z = self.upsample_latent(z.reshape(-1, 1, 64, 64))
            #z = z.reshape(-1, 1, 128, 128)
            # Concatenate the feature map (output of the UNet) and the sample taken from the latent space
            feature_map = torch.cat((feature_map, z), dim=self.channel_axis)
            x = self.layers(feature_map)
            y = self.last_layer(x)
            if use_softmax:
                y = self.activation(y)
            return y

class PiononoModelProb(nn.Module):
    """
    The implementation of the Pionono Model. It consists of a segmentation backbone, probabilistic latent variable and
    segmentation head.
    """

    def __init__(self, input_channels=3, num_classes=1, annotators=6, gold_annotators=[0], latent_dim=4,
                 z_prior_mu=0.0, z_prior_sigma=2.0, z_posterior_init_sigma=8.0, no_head_layers=3, head_kernelsize=1,
                 head_dilation=1, kl_factor=1.0, reg_factor=0.1, mc_samples=5):
        super(PiononoModelProb, self).__init__()
        #latent_dim = len(annotators)
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.annotators = annotators
        self.gold_annotators = gold_annotators
        self.no_head_layers = no_head_layers
        self.head_kernelsize = head_kernelsize
        self.head_dilation = head_dilation
        self.kl_factor = kl_factor
        self.reg_factor = reg_factor
        self.train_mc_samples = mc_samples
        self.test_mc_samples = 20
        self.unet = UnetHeadless(input_channels).to(device)
        self.z = LatentVariable(len(annotators), latent_dim, prior_mu_value=z_prior_mu, prior_sigma_value=z_prior_sigma,
                                z_posterior_init_sigma=z_posterior_init_sigma).to(device)
        self.head = PiononoHead(16, self.latent_dim, self.input_channels, self.num_classes,
                                self.no_head_layers, self.head_kernelsize, self.head_dilation, use_tile=True).to(device)
        self.phase = 'segmentation'
        self.name = 'PiononoModelProb'

    def forward(self, patch):
        """
        Get feature maps.
        """
        self.unet_features = self.unet.forward(patch)

    def map_annotators_to_correct_id(self, annotator_ids: torch.tensor, annotator_list:list = None):
        new_ids = torch.zeros_like(annotator_ids).to(device)
        for a in range(len(annotator_ids)):
            id_corresponds = (annotator_list[int(annotator_ids[a])] == np.array(self.annotators))
            if not np.any(id_corresponds):
                raise Exception('Annotator has no corresponding distribution. Annotator: ' + str(annotator_list[int(annotator_ids[a])]))
            new_ids[a] = torch.nonzero(torch.tensor(annotator_list[int(annotator_ids[a])] == np.array(self.annotators)))[0][0]
        return new_ids

    def sample(self, use_z_mean: bool, annotator_ids: torch.tensor, annotator_list: list = None, use_softmax=True):
        """
        Get sample of output distribution. Annotator list defines the distributions (q|r) that are used.
        """
        if annotator_list is not None:
            annotator_ids = self.map_annotators_to_correct_id(annotator_ids, annotator_list)

        if use_z_mean == False:
            z = self.z.forward(annotator_ids, sample=True)
        else:
            z = self.z.forward(annotator_ids, sample=False)
        pred = self.head.forward(self.unet_features, z, annotator_ids, use_softmax)

        return pred

    def prior_sample(self, use_z_mean: bool, annotator_ids: torch.tensor, annotator_list: list = None, use_softmax=True):
        """
        Get sample of output distribution. Annotator list defines the distributions (q|r) that are used.
        """
        if annotator_list is not None:
            annotator_ids = self.map_annotators_to_correct_id(annotator_ids, annotator_list)

        if use_z_mean == False:
            z = self.z.prior_forward(annotator_ids, sample=True)
        else:
            z = self.z.prior_forward(annotator_ids, sample=False)
        pred = self.head.forward(self.unet_features, z, use_softmax)

        return pred

    def get_gold_predictions(self):
        """
        Get gold predictions (based on the gold distribution).
        """
        if len(self.gold_annotators) == 1:
            annotator = torch.ones(self.unet_features.shape[0]).to(device) * self.gold_annotators[0]
            mean, std = self.mc_sampling(annotator, use_softmax=True)
        else:
            shape = [self.train_mc_samples * len(self.gold_annotators), self.unet_features.shape[0], self.num_classes,
                     self.unet_features.shape[-2], self.unet_features.shape[-1]]
            samples = torch.zeros(shape).to(device)
            for a in range(len(self.gold_annotators)):
                for i in range(self.train_mc_samples):
                    annotator_ids = torch.ones(self.unet_features.shape[0]).to(device) * self.gold_annotators[a]
                    samples[(a * self.train_mc_samples) + i] = self.sample(use_z_mean=False,
                                                                           annotator_ids=annotator_ids,
                                                                           use_softmax=True)
            mean = torch.mean(samples, dim=0)
            std = torch.std(samples, dim=0)
        return mean, std

    def mc_sampling(self, annotator: torch.tensor = None, use_softmax=True):
        """
        Monte-Carlo sampling to get mean and std of output distribution.
        """
        if self.training:
            mc_samples = self.train_mc_samples
        else:
            mc_samples = self.test_mc_samples
        shape = [mc_samples, annotator.shape[0], self.num_classes, self.unet_features.shape[-2], self.unet_features.shape[-1]]
        samples = torch.zeros(shape).to(device)
        for i in range(mc_samples):
            samples[i] = self.sample(use_z_mean=False, annotator_ids=annotator, use_softmax=use_softmax)
        mean = torch.mean(samples, dim=0)
        std = torch.std(samples, dim=0)
        return mean, std

    def mc_prior_sampling(self, annotator: torch.tensor = None, use_softmax=True):
        if self.training:
            mc_samples = self.train_mc_samples
        else:
            mc_samples = self.test_mc_samples
        shape = [mc_samples, annotator.shape[0], self.num_classes, self.unet_features.shape[-2], self.unet_features.shape[-1]]
        samples = torch.zeros(shape).to(device)
        for i in range(mc_samples):
            samples[i] = self.prior_sample(use_z_mean=False, annotator_ids=annotator, use_softmax=use_softmax)
        mean = torch.mean(samples, dim=0)
        std = torch.std(samples, dim=0)
        return mean, std        

    def elbo(self, labels: torch.tensor, loss_fct, annotator: torch.tensor, image: torch.tensor):
        """
        Calculate the evidence lower bound of the log-likelihood of P(Y|X)
        """
        # self.preds = self.sample(use_z_mean=False, annotator=annotator)
        self.preds, _ = self.mc_sampling(annotator=annotator, use_softmax=False)
        self.log_likelihood_loss = loss_fct(self.preds, labels)
        #self.kl_loss = self.z.get_kl_loss(annotator) * self.kl_factor
        self.kl_loss = self.head.get_kl_loss(annotator, self.head.feature_map) * self.kl_factor
        self.log_likelihood_loss += (self.z.get_class_ce_loss(annotator))

        self.log_likelihood_loss += (self.head.get_reconstruction_loss(image, annotator))

        return -(self.log_likelihood_loss + self.kl_loss)

    def combined_loss(self, labels, loss_fct, annotator, image):
        """
        Combine ELBO with regularization of deep network weights.
        """
        elbo = self.elbo(labels, loss_fct=loss_fct, annotator=annotator, image=image)
        self.reg_loss = l2_regularisation(self.head.layers) * self.reg_factor
        loss = -elbo + self.reg_loss
        return loss

    def train_step(self, images, labels, loss_fct, ann_ids):
        """
        Make one train step, returning loss and predictions.
        """
        self.forward(images)
        loss = self.combined_loss(labels, loss_fct, ann_ids, images)
        y_pred = self.preds

        return loss, y_pred
    
    def val_step(self, images):
        """
        Make one train step, returning loss and predictions.
        """
        self.forward(images)
        y_pred = []
        meta = torch.ones([images.shape[0]])
        for i in range(4):
            y_pred_case, _ = self.mc_sampling(annotator=meta*i, use_softmax=False)
            y_pred.append(y_pred_case)
        y_pred = torch.cat(y_pred, dim=1)
        return y_pred
    
    def test_step(self, images):
        """
        Make one train step, returning loss and predictions.
        """
        self.forward(images)
        y_pred = []
        meta = torch.ones([images.shape[0]])
        for i in range(4):
            y_pred_case, _ = self.mc_sampling(annotator=meta*i, use_softmax=False)
            y_pred.append(y_pred_case)
        y_pred = torch.cat(y_pred, dim=1)
        return y_pred
    
    def sample_test(self, images):
        """
        sample the prior distribution
        """
        self.forward(images)
        y_pred = []
        meta = torch.ones([images.shape[0]])
        for i in range(4):
            y_pred_case, _ = self.mc_prior_sampling(annotator=meta*i, use_softmax=True)
            y_pred.append(y_pred_case)
        y_pred = torch.cat(y_pred, dim=1)
        return y_pred
