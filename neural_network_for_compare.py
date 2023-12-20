import torch
import torch.nn.functional as F
from basic_parameter import *
from copy import deepcopy as DC
from util.util_function import sparse_degree


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
        if np.any(PSI) == None:
            self.theta = nn.Parameter(torch.ones((N, G), dtype=torch.float32), requires_grad=True)
            nn.init.uniform_(self.theta, a = 0, b=2*np.pi)
            self.alpha = nn.Parameter(torch.ones((N, G), dtype=torch.float32), requires_grad=True)
            nn.init.uniform_(self.alpha)

        else:
            PSI = torch.from_numpy(PSI)
            PSI = complex_feature_trasnsmission(PSI)
            self.PSI = nn.Parameter(PSI, requires_grad=True)

        # ISTA Stepsizes eta = 1/L
        self.etas = nn.Parameter(1e-4*torch.ones(T + 1, 1, 1), requires_grad=True)
        self.gammas = nn.Parameter(torch.ones(T + 1, 1, 1), requires_grad=True)
        # Initialization
        self.reinit_num = 0  # Number of re-initializations

    def _shrink(self, x, eta):
        return eta * F.softshrink(x / eta, lambd=self.lambd)

    def _shrink2(self, x, eta):
        return eta * F.tanhshrink(x/ eta)

    def _pooling(self, x):
        xabs = x[0]**2 + x[1]**2
        xabs = xabs.unsqueeze(0).expand(2, -1, -1)
        xmax = xabs.max(dim=-1, keepdim=True).values.expand(2, -1, self.G)
        x[xabs < xmax] = 0
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
        self.alpha_sigmoid = F.sigmoid(self.alpha)/np.sqrt(N)
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
        x = C(H, self.PSI)
        x = self._pooling(x)
        return x, C(x, self._H(self.PSI))



    def supervised_learning_sparse_domain(self, x, label):
        label_ = C(label, self.PSI)
        return huber_loss(x[0], label_ * 1e3) + huber_loss(x[1], label_ * 1e3)

    def supervised_learning_corr(self, x, label):
        norm_label = self._norm_2(label)
        corr = self._diag(C(label, self._H(x)))
        corr = torch.sqrt(corr[0] ** 2 + corr[1] ** 2)/norm_label
        return torch.sum(corr)

    def constrain(self):
        corr = C(self._H(self.PSI), self.PSI)
        corr = torch.sqrt(corr[0]**2 + corr[1]**2)
        return F.mse_loss(corr, self.I)






class Dictionary_Learning_hierarchical(nn.Module):


    def __init__(self, N, G, PSI = None,  W_initial = None, T=16, lambd=1):
        '''
        :param N:  the array size
        :param n:  the grid siaze
        :param W_initial： the initial of W

        :param T:  total layer time
        :param lambd: parametet
        '''

        super(Dictionary_Learning_hierarchical, self).__init__()


        self.N, self.G = N, G
        layer = np.log2(self.G)
        self.T = T  # ISTA Iterations
        self.lambd = lambd  # Lagrangian Multiplier

        self.I = nn.Parameter(torch.eye(self.G), requires_grad=False)
        if np.any(PSI) == None:
            par_list = []
            for i in range(layer):
                PSI = nn.Parameter(torch.ones((2, N, np.power(2, (i+1))), dtype=torch.float32), requires_grad=True)
                nn.init.xavier_normal_(PSI)
                par_list.append(PSI)
            self.PSI = nn.ParameterList(par_list)

        else:
            PSI = torch.from_numpy(PSI)
            PSI = complex_feature_trasnsmission(PSI)
            par_list = []
            for i in range(layer):
                number_of_beams = np.power(2, (i+1))
                beams_area = self.G / np.power(2, (i+1))
                beam = torch.zeros((2, self.N, number_of_beams))
                for j in range(number_of_beams):
                    range_area = list(range(j*beams_area,(j+1)*beams_area))
                    beam[2, :, ] = torch.sum(PSI[:,:, range_area], dim=-1)
                par_list.append(beam)

            self.PSI = nn.Parameter(PSI, requires_grad=True)


        # self.A.data = DC(We)
        # self.B.data = DC(Wd)
        # ISTA Stepsizes eta = 1/L
        self.etas = nn.Parameter(1e-4*torch.ones(T + 1, 1, 1), requires_grad=True)
        self.gammas = nn.Parameter(torch.ones(T + 1, 1, 1), requires_grad=True)
        # Initialization
        self.reinit_num = 0  # Number of re-initializations

    def _shrink(self, x, eta):
        return eta * F.softshrink(x / eta, lambd=self.lambd)

    def _shrink2(self, x, eta):
        return eta * F.tanhshrink(x/ eta)

    def _pooling(self, x):
        xabs = x[0]**2 + x[1]**2
        xabs = xabs.unsqueeze(0).expand(2, -1, -1)
        xmax = xabs.max(dim=-1, keepdim=True).values.expand(2, -1, self.G)
        x[xabs < xmax] = 0
        return x

    def _T(self, H):
        return H.transpose(1, 2)

    def _H(self, H):
        conj_matrix = torch.ones_like(H)
        conj_matrix[1] = -1
        return (conj_matrix*H).transpose(1, 2)

    def forward(self, H):
        '''
        :param y: the obversation signal with the dim BN, N ,1
        :return:
               x: recoveried sparse signal with the dim BN, G ,1
        '''
        H = complex_feature_trasnsmission(H)

        x = torch.zeros((2, batchsize, self.G))
        x = x.cuda()

        for i, par in enumerate(self.PSI):
            x = x - self.gammas[i, :, :] * (x - C(H, par))
            # x = self._shrink(x, self.etas[i, :, :])
            x = self._pooling(x)


        return x, C(x, self._H(self.PSI))


    def constrain(self):
        corr = C(self._H(self.PSI), self.PSI)
        corr = torch.sqrt(corr[0]**2 + corr[1]**2)
        return F.mse_loss(corr, self.I)