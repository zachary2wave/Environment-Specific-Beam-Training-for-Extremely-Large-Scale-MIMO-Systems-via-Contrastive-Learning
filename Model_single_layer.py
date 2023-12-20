import torch
import torch.nn.functional as F
from basic_parameter import *



class Dictionary_Learning(nn.Module):
    def __init__(self, N, G, PSI = None,  W_initial = None, T=1, lambd=1):
        '''
        :param N:  the array size
        :param n:  the grid siaze
        :param W_initial： the initial of W

        :param T:  total layer time
        :param lambd: parametet
        '''

        super(Dictionary_Learning, self).__init__()


        self.N, self.G = N, G

        self.T = T  # ISTA Iterations
        self.lambd = lambd  # Lagrangian Multiplier

        self.I = nn.Parameter(torch.eye(self.G), requires_grad=False)
        self.I_in = nn.Parameter(-1*torch.eye(batchsize), requires_grad=False)
        self.obI = nn.Parameter(torch.ones([batchsize, batchsize]) - torch.eye(batchsize), requires_grad=False)


        if np.any(PSI) == None:
            self.theta = nn.Parameter(torch.ones((N, G), dtype=torch.float32), requires_grad=True)
            nn.init.uniform_(self.theta, a = 0, b=2*np.pi)
            self.alpha = nn.Parameter(torch.ones((N, G), dtype=torch.float32), requires_grad=True)
            nn.init.uniform_(self.alpha)

        else:
            # PSI = torch.from_numpy(PSI)
            # PSI = complex_feature_trasnsmission(PSI)
            # self.PSI = nn.Parameter(PSI, requires_grad=True)
            theta = np.angle(PSI)
            theta = torch.from_numpy(theta)
            self.theta = nn.Parameter(theta, requires_grad=True)
            self.alpha = nn.Parameter(100*torch.ones((N, G), dtype=torch.float32), requires_grad=True)

        # Momentum Contrast
        self.update_direction = nn.Parameter(torch.ones(N, G), requires_grad=False)
        self.momentun_parameter = 0.1


        # hierarchical layer
        self.PHI = self.obtain_psi()
        pass



    def _shrink(self, x, eta):
        return eta * F.softshrink(x / eta, lambd=self.lambd)

    def _shrink2(self, x, eta):
        return eta * F.tanhshrink(x/ eta)

    def _pooling(self, x):
        xabs = x[0]**2 + x[1]**2
        xabs = xabs.unsqueeze(0).expand(2, -1, -1)
        xmax = xabs.max(dim=-1, keepdim=True).values.expand(2, -1, self.G)
        xmask = (xabs == xmax).to(dtype=torch.int32)
        x = xmask*x
        return x

    def _T(self, H):
        return H.transpose(1, 2)

    def _H(self, H):
        conj_matrix = torch.ones_like(H)
        conj_matrix[1] = -1
        return (conj_matrix*H).transpose(1, 2)

    def _conj(self,H):
        conj_matrix = torch.ones_like(H)
        conj_matrix[1] = -1
        return (conj_matrix * H)

    def _diag(self, H):
        return torch.diagonal(H, dim1=1, dim2=2)

    def _norm_2(self, H):
        return torch.sum(torch.sqrt(H[0] ** 2 + H[1] ** 2), dim=1)

    def _dropout(self, x):
        drop_prob = torch.rand(self.G)
        drop_mask = torch.ones_like(x)
        return drop_mask*x

    def obtain_psi(self):
        self.alpha_sigmoid = F.tanh(self.alpha)/np.sqrt(N)
        PSI_real = self.alpha_sigmoid * torch.cos(self.theta)
        PSI_imag = self.alpha_sigmoid * torch.sin(self.theta)
        return torch.cat([PSI_real.unsqueeze(0), PSI_imag.unsqueeze(0)], dim=0)

    def forward(self, H):
        '''
        :param y: the obversation signal with the dim BN, N ,1
        :return:
               x: recoveried sparse signal with the dim BN, G ,1
        '''
        H = complex_feature_trasnsmission(H)
        self.alpha_sigmoid = F.tanh(self.alpha)/np.sqrt(N)
        PSI_real = self.alpha_sigmoid * torch.cos(self.theta)
        PSI_imag = self.alpha_sigmoid * torch.sin(self.theta)
        PSI = torch.cat([PSI_real.unsqueeze(0), PSI_imag.unsqueeze(0)], dim=0)
        x = C(H, PSI)
        x = self._pooling(x)
        return x, C(x, self._H(PSI))


    def unsupervised_learning(self, x, label):
        norm_label = self._norm_2(label)
        xcorr = self._diag(C(x, self._H(x)))
        outcome = torch.sqrt(xcorr[0] ** 2 + xcorr[1] ** 2)/norm_label
        return torch.sum(torch.max(outcome, dim=-1).values)

    def unsupervised_learning_corr(self, label):
        H = complex_feature_trasnsmission(label)
        # NormH = self._norm_2(H)
        self.alpha_sigmoid = F.tanh(self.alpha)/np.sqrt(N)
        PSI_real = self.alpha_sigmoid * torch.cos(self.theta)
        PSI_imag = self.alpha_sigmoid * torch.sin(self.theta)
        PSI = torch.cat([PSI_real.unsqueeze(0), PSI_imag.unsqueeze(0)], dim=0)

        PHI_orthogonal = []

        for phi_indix in range(self.G):
            ori = PSI[:, :, phi_indix].unsqueeze(-1)
            for i in range(phi_indix):
                 ori = ori - C(self._H(ori), PHI_orthogonal[i])*PHI_orthogonal[i]
            ori_norm = C(self._H(ori), ori).squeeze()[0]
            PHI_orthogonal.append(ori/ori_norm)
        W = torch.cat(PHI_orthogonal, dim=-1)


        x = C(H, W)
        x = self._pooling(x)
        norm_label = torch.diagonal(label.mm(torch.conj(label.T)))
        xcorr = self._diag(C(x, self._H(x)))
        outcome = torch.sqrt(xcorr[0] ** 2 + xcorr[1] ** 2)/norm_label
        return torch.sum(torch.abs(outcome))


    def contract_learning(self, los_path, label_H):
        label_H = label_H * 1e3
        # label_H_conj = torch.conj(label_H)
        # coefficient = 1/torch.abs(torch.diagonal(torch.matmul(label_H_conj, label_H.T)))
        # coefficient = coefficient.repeat(batchsize, 1)

        coefficient = torch.ones([batchsize, batchsize])
        los_path_conj = torch.conj(label_H)
        coefficient_mask = torch.abs(torch.matmul(los_path_conj, label_H.T))
        coefficient_mask_diag = torch.diagonal(coefficient_mask)
        coefficient_mask = coefficient_mask/coefficient_mask_diag
        coefficient_mask = (coefficient_mask-0.49)
        coefficient[coefficient_mask > 0] = -1
        coefficient = coefficient.cuda()

        H = complex_feature_trasnsmission(label_H)
        # NormH = self._norm_2(H)
        self.alpha_sigmoid = F.tanh(self.alpha)/np.sqrt(N)
        PSI_real = self.alpha_sigmoid * torch.cos(self.theta)
        PSI_imag = self.alpha_sigmoid * torch.sin(self.theta)
        PSI = torch.cat([PSI_real.unsqueeze(0), PSI_imag.unsqueeze(0)], dim=0)

        # PHI_orthogonal = []
        #
        # for phi_indix in range(self.G):
        #     ori = PSI[:, :, phi_indix].unsqueeze(-1)
        #     for i in range(phi_indix):
        #          ori = ori - C(self._H(ori), PHI_orthogonal[i])*PHI_orthogonal[i]
        #     ori_norm = C(self._H(ori), ori).squeeze()[0]
        #     PHI_orthogonal.append(ori/ori_norm)
        # W = torch.cat(PHI_orthogonal, dim=-1)

        X = C(H, PSI)
        X = torch.sqrt(X[0] ** 2 + X[1] ** 2)
        # Xselected, X_indice = torch.max(X, dim=-1)

        X_probablity = F.softmax(X*100, dim =-1)

        xcorr = torch.matmul(X_probablity, X_probablity.T)
        # self_matrix = torch.diag(Xselected * Xselected)

        I = torch.eye(batchsize)
        I = I.cuda()
        Ones = torch.ones([batchsize,batchsize])
        Ones = Ones.cuda()

        contractloss = coefficient * ((Ones-I) * xcorr)
        #

        return torch.sum(contractloss)

    def supervised_learning_sparse_domain(self, x, label):
        self.alpha_sigmoid = F.tanh(self.alpha)
        PSI_real = self.alpha_sigmoid * torch.cos(self.theta)
        PSI_imag = self.alpha_sigmoid * torch.sin(self.theta)
        PSI = torch.cat([PSI_real.unsqueeze(0), PSI_imag.unsqueeze(0)], dim=0)
        label_ = C(label, PSI)
        return huber_loss(x[0], label_ * 1e3) + huber_loss(x[1], label_ * 1e3)

    def supervised_learning_real_domain(self, x, label):
        self.alpha_sigmoid = F.tanh(self.alpha)
        PSI_real = self.alpha_sigmoid * torch.cos(self.theta)
        PSI_imag = self.alpha_sigmoid * torch.sin(self.theta)
        PSI = torch.cat([PSI_real.unsqueeze(0), PSI_imag.unsqueeze(0)], dim=0)
        label_ = C(label, PSI)
        return huber_loss(x[0], label_ * 1e3) + huber_loss(x[1], label_ * 1e3)

    def constrain(self):
        corr = C(self._H(self.PSI), self.PSI)
        corr = torch.sqrt(corr[0]**2 + corr[1]**2)
        return F.mse_loss(corr, self.I)




class Dictionary_Learning_first_stage(nn.Module):
    def __init__(self, N, G, PSI = None, T=1, lambd=1):
        '''
        :param N:  the array size
        :param n:  the grid siaze
        :param W_initial： the initial of W

        :param T:  total layer time
        :param lambd: parametet
        '''

        super(Dictionary_Learning_first_stage, self).__init__()


        self.N, self.G = N, G

        self.T = T  # ISTA Iterations
        self.lambd = lambd  # Lagrangian Multiplier

        self.I = nn.Parameter(torch.eye(self.G), requires_grad=False)
        self.I_in = nn.Parameter(-1*torch.eye(batchsize), requires_grad=False)
        self.obI = nn.Parameter(torch.ones([batchsize, batchsize]) - torch.eye(batchsize), requires_grad=False)


        if np.any(PSI) == None:
            self.theta = nn.Parameter(torch.ones((N, G), dtype=torch.float32), requires_grad=True)
            nn.init.uniform_(self.theta, a = 0, b=2*np.pi)
            self.alpha = nn.Parameter(torch.ones((N, G), dtype=torch.float32), requires_grad=True)
            nn.init.uniform_(self.alpha)




        else:
            # PSI = torch.from_numpy(PSI)
            # PSI = complex_feature_trasnsmission(PSI)
            # self.PSI = nn.Parameter(PSI, requires_grad=True)
            theta = np.angle(PSI)
            theta = torch.from_numpy(theta)
            self.kappa = nn.Parameter(theta, requires_grad=True)
            self.alpha = nn.Parameter(100*torch.ones((N, G), dtype=torch.float32), requires_grad=True)

        # Momentum Contrast
        self.update_direction = nn.Parameter(torch.ones(N, G), requires_grad=False)
        self.momentun_parameter = 0.1


        # hierarchical layer
        self.PHI = self.obtain_psi()
        pass



    def _shrink(self, x, eta):
        return eta * F.softshrink(x / eta, lambd=self.lambd)

    def _shrink2(self, x, eta):
        return eta * F.tanhshrink(x/ eta)

    def _pooling(self, x):
        xabs = x[0]**2 + x[1]**2
        xabs = xabs.unsqueeze(0).expand(2, -1, -1)
        xmax = xabs.max(dim=-1, keepdim=True).values.expand(2, -1, self.G)
        xmask = (xabs == xmax).to(dtype=torch.int32)
        x = xmask*x
        return x

    def _T(self, H):
        return H.transpose(1, 2)

    def _H(self, H):
        conj_matrix = torch.ones_like(H)
        conj_matrix[1] = -1
        return (conj_matrix*H).transpose(1, 2)

    def _conj(self,H):
        conj_matrix = torch.ones_like(H)
        conj_matrix[1] = -1
        return (conj_matrix * H)

    def _diag(self, H):
        return torch.diagonal(H, dim1=1, dim2=2)

    def _norm_2(self, H):
        return torch.sum(torch.sqrt(H[0] ** 2 + H[1] ** 2), dim=1)

    def _dropout(self, x):
        drop_prob = torch.rand(self.G)
        drop_mask = torch.ones_like(x)
        return drop_mask*x

    def obtain_psi(self):
        self.alpha_sigmoid = F.tanh(self.alpha)/np.sqrt(N)
        PSI_real = self.alpha_sigmoid * torch.cos(self.theta)
        PSI_imag = self.alpha_sigmoid * torch.sin(self.theta)
        return torch.cat([PSI_real.unsqueeze(0), PSI_imag.unsqueeze(0)], dim=0)

    def forward(self, H):
        '''
        :param y: the obversation signal with the dim BN, N ,1
        :return:
               x: recoveried sparse signal with the dim BN, G ,1
        '''
        H = complex_feature_trasnsmission(H)
        self.alpha_sigmoid = F.tanh(self.alpha)/np.sqrt(N)
        PSI_real = self.alpha_sigmoid * torch.cos(self.theta)
        PSI_imag = self.alpha_sigmoid * torch.sin(self.theta)
        PSI = torch.cat([PSI_real.unsqueeze(0), PSI_imag.unsqueeze(0)], dim=0)
        x = C(H, PSI)
        return x


class Dictionary_Learning_second_stage(nn.Module):
    def __init__(self, N, G, PSI = None, T=1, lambd=1):
        '''
        :param N:  the array size
        :param n:  the grid siaze
        :param W_initial： the initial of W

        :param T:  total layer time
        :param lambd: parametet
        '''

        super(Dictionary_Learning_second_stage, self).__init__()


        self.N, self.G = N, G

        self.T = T  # ISTA Iterations
        self.lambd = lambd  # Lagrangian Multiplier

        self.I = nn.Parameter(torch.eye(self.G), requires_grad=False)
        self.I_in = nn.Parameter(-1*torch.eye(batchsize), requires_grad=False)
        self.obI = nn.Parameter(torch.ones([batchsize, batchsize]) - torch.eye(batchsize), requires_grad=False)


        if np.any(PSI) == None:
            self.theta = nn.Parameter(torch.ones((N, G), dtype=torch.float32), requires_grad=True)
            nn.init.uniform_(self.theta, a = 0, b=2*np.pi)
            self.alpha = nn.Parameter(torch.ones((N, G), dtype=torch.float32), requires_grad=True)
            nn.init.uniform_(self.alpha)




        else:
            # PSI = torch.from_numpy(PSI)
            # PSI = complex_feature_trasnsmission(PSI)
            # self.PSI = nn.Parameter(PSI, requires_grad=True)
            theta = np.angle(PSI)
            theta = torch.from_numpy(theta)
            self.kappa = nn.Parameter(theta, requires_grad=True)
            self.alpha = nn.Parameter(100*torch.ones((N, G), dtype=torch.float32), requires_grad=True)

        # Momentum Contrast
        self.update_direction = nn.Parameter(torch.ones(N, G), requires_grad=False)
        self.momentun_parameter = 0.1


        # hierarchical layer
        self.PHI = self.obtain_psi()
        pass



    def _shrink(self, x, eta):
        return eta * F.softshrink(x / eta, lambd=self.lambd)

    def _shrink2(self, x, eta):
        return eta * F.tanhshrink(x/ eta)

    def _pooling(self, x):
        xabs = x[0]**2 + x[1]**2
        xabs = xabs.unsqueeze(0).expand(2, -1, -1)
        xmax = xabs.max(dim=-1, keepdim=True).values.expand(2, -1, self.G)
        xmask = (xabs == xmax).to(dtype=torch.int32)
        x = xmask*x
        return x

    def _T(self, H):
        return H.transpose(1, 2)

    def _H(self, H):
        conj_matrix = torch.ones_like(H)
        conj_matrix[1] = -1
        return (conj_matrix*H).transpose(1, 2)

    def _conj(self,H):
        conj_matrix = torch.ones_like(H)
        conj_matrix[1] = -1
        return (conj_matrix * H)

    def _diag(self, H):
        return torch.diagonal(H, dim1=1, dim2=2)

    def _norm_2(self, H):
        return torch.sum(torch.sqrt(H[0] ** 2 + H[1] ** 2), dim=1)

    def _dropout(self, x):
        drop_prob = torch.rand(self.G)
        drop_mask = torch.ones_like(x)
        return drop_mask*x

    def obtain_psi(self):
        self.alpha_sigmoid = F.tanh(self.alpha)/np.sqrt(N)
        PSI_real = self.alpha_sigmoid * torch.cos(self.theta)
        PSI_imag = self.alpha_sigmoid * torch.sin(self.theta)
        return torch.cat([PSI_real.unsqueeze(0), PSI_imag.unsqueeze(0)], dim=0)

    def forward(self, H):
        '''
        :param y: the obversation signal with the dim BN, N ,1
        :return:
               x: recoveried sparse signal with the dim BN, G ,1
        '''
        H = complex_feature_trasnsmission(H)
        self.alpha_sigmoid = F.tanh(self.alpha)/np.sqrt(N)
        PSI_real = self.alpha_sigmoid * torch.cos(self.theta)
        PSI_imag = self.alpha_sigmoid * torch.sin(self.theta)
        PSI = torch.cat([PSI_real.unsqueeze(0), PSI_imag.unsqueeze(0)], dim=0)
        x = C(H, PSI)
        return x

    def contract_learning(self, label_H):
        label_H = label_H * 1e3
        batch_size = label_H.shape[0]
        coefficient = torch.ones([batch_size, batch_size])
        los_path_conj = torch.conj(label_H)
        coefficient_mask = torch.abs(torch.matmul(los_path_conj, label_H.T))
        coefficient_mask_diag = torch.diagonal(coefficient_mask)
        coefficient_mask = coefficient_mask/coefficient_mask_diag
        coefficient_mask = (coefficient_mask-0.6)
        coefficient[coefficient_mask > 0] = -1
        coefficient = coefficient.cuda()

        H = complex_feature_trasnsmission(label_H)
        # NormH = self._norm_2(H)
        self.alpha_sigmoid = F.tanh(self.alpha)/np.sqrt(N)
        PSI_real = self.alpha_sigmoid * torch.cos(self.theta)
        PSI_imag = self.alpha_sigmoid * torch.sin(self.theta)
        PSI = torch.cat([PSI_real.unsqueeze(0), PSI_imag.unsqueeze(0)], dim=0)

        X = C(H, PSI)
        X = torch.sqrt(X[0] ** 2 + X[1] ** 2)

        X_probablity = F.softmax(X/1e5, dim =-1)

        xcorr = torch.matmul(X_probablity, X_probablity.T)

        I = torch.eye(batch_size)
        I = I.cuda()
        Ones = torch.ones([batch_size,batch_size])
        Ones = Ones.cuda()

        contractloss = coefficient * (Ones-I) * xcorr
        #

        return torch.sum(contractloss)


class Dictionary_Learning_second_stage_V2(nn.Module):
    def __init__(self, N, G, T=1, lambd=1):
        '''
        :param N:  the array size
        :param n:  the grid siaze
        :param W_initial： the initial of W

        :param T:  total layer time
        :param lambd: parametet
        '''

        super(Dictionary_Learning_second_stage_V2, self).__init__()


        self.N, self.G = N, G

        self.T = T  # ISTA Iterations
        self.lambd = lambd  # Lagrangian Multiplier

        self.I = nn.Parameter(torch.eye(self.G), requires_grad=False)
        self.I_in = nn.Parameter(-1*torch.eye(batchsize), requires_grad=False)
        self.obI = nn.Parameter(torch.ones([batchsize, batchsize]) - torch.eye(batchsize), requires_grad=False)



        self.b = nn.Parameter(torch.ones((G, 1), dtype=torch.float32), requires_grad=True)
        nn.init.uniform_(self.b, a = 2, b=100)
        self.a = nn.Parameter(torch.ones((G, 1), dtype=torch.float32), requires_grad=True)
        nn.init.uniform_(self.a, a = -1, b=1)


        # Momentum Contrast
        self.update_direction = nn.Parameter(torch.ones(N, G), requires_grad=False)
        self.momentun_parameter = 0.1


        # # hierarchical layer
        # self.PHI = self.obtain_psi()
        # pass



    def _shrink(self, x, eta):
        return eta * F.softshrink(x / eta, lambd=self.lambd)

    def _shrink2(self, x, eta):
        return eta * F.tanhshrink(x/ eta)

    def _pooling(self, x):
        xabs = x[0]**2 + x[1]**2
        xabs = xabs.unsqueeze(0).expand(2, -1, -1)
        xmax = xabs.max(dim=-1, keepdim=True).values.expand(2, -1, self.G)
        xmask = (xabs == xmax).to(dtype=torch.int32)
        x = xmask*x
        return x

    def _T(self, H):
        return H.transpose(1, 2)

    def _H(self, H):
        conj_matrix = torch.ones_like(H)
        conj_matrix[1] = -1
        return (conj_matrix*H).transpose(1, 2)

    def _conj(self,H):
        conj_matrix = torch.ones_like(H)
        conj_matrix[1] = -1
        return (conj_matrix * H)

    def _diag(self, H):
        return torch.diagonal(H, dim1=1, dim2=2)

    def _norm_2(self, H):
        return torch.sum(torch.sqrt(H[0] ** 2 + H[1] ** 2), dim=1)

    def _dropout(self, x):
        drop_prob = torch.rand(self.G)
        drop_mask = torch.ones_like(x)
        return drop_mask*x

    def obtain_psi(self):
        theta = F.tanh(self.a)
        r = F.relu(self.b)+2
        array = torch.range(0, self.N-1) * d_interval_x
        array = array.unsqueeze(-1).T
        array = array.cuda()
        phase = (theta + torch.matmul((1-theta**2)/2/r, array))*array
        PSI_real = torch.cos(-k*phase)
        PSI_imag = torch.sin(-k*phase)
        return torch.cat([PSI_real.unsqueeze(0), PSI_imag.unsqueeze(0)], dim=0)


    def forward(self, H):
        '''
        :param y: the obversation signal with the dim BN, N ,1
        :return:
               x: recoveried sparse signal with the dim BN, G ,1
        '''
        H = complex_feature_trasnsmission(H)
        theta = F.tanh(self.a)
        r = F.relu(self.b)+2
        array = torch.range(0, self.N-1) * d_interval_x
        array = array.unsqueeze(-1).T
        phase = (theta + torch.matmul((1-theta**2)/2/r, array))*array
        PSI_real = torch.cos(-k*phase)
        PSI_imag = torch.sin(-k*phase)
        PSI = torch.cat([PSI_real.unsqueeze(0), PSI_imag.unsqueeze(0)], dim=0)
        PSI = self._T(PSI)
        x = C(H, PSI)
        return x

    def contract_learning(self, label_H):
        label_H = label_H * 1e3
        batch_size = label_H.shape[0]
        coefficient = torch.ones([batch_size, batch_size])
        los_path_conj = torch.conj(label_H)
        coefficient_mask = torch.abs(torch.matmul(los_path_conj, label_H.T))
        coefficient_mask_diag = torch.diagonal(coefficient_mask)
        coefficient_mask = coefficient_mask/coefficient_mask_diag
        coefficient_mask = (coefficient_mask-0.9)
        coefficient[coefficient_mask > 0] = -1
        coefficient = coefficient.cuda()

        H = complex_feature_trasnsmission(label_H)
        # NormH = self._norm_2(H)
        theta = F.tanh(self.a)
        r = F.relu(self.b)+2
        array = torch.range(0, self.N-1) * d_interval_x
        array = array.unsqueeze(-1).T
        array = array.cuda()
        phase = (theta + torch.matmul((1-theta**2)/2/r, array))*array
        PSI_real = torch.cos(-k*phase)
        PSI_imag = torch.sin(-k*phase)
        PSI = torch.cat([PSI_real.unsqueeze(0), PSI_imag.unsqueeze(0)], dim=0)
        PSI = self._T(PSI)
        X = C(H, PSI)
        X = torch.sqrt(X[0] ** 2 + X[1] ** 2)

        X_probablity = F.softmax(X/10, dim =-1)

        xcorr = torch.matmul(X_probablity, X_probablity.T)

        I = torch.eye(batch_size)
        I = I.cuda()
        Ones = torch.ones([batch_size,batch_size])
        Ones = Ones.cuda()

        contractloss = coefficient * (Ones-I) * xcorr
        #

        return torch.sum(contractloss)