import torch
import torch.nn.functional as F
import timeit
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt


class Net(torch.nn.Module):

    def __init__(self, n_input, n_hidden, n_output, lqr):
        super(Net, self).__init__()
        torch.manual_seed(2)
        self.layer1 = torch.nn.Linear(n_input, n_hidden)
        self.layer2 = torch.nn.Linear(n_hidden, n_output)
        self.control = torch.nn.Linear(n_input, 1, bias=False)
        self.control.weight = torch.nn.Parameter(lqr)

    def forward(self, x):
        sigmoid = torch.nn.Tanh()
        h_1 = sigmoid(self.layer1(x))
        out = sigmoid(self.layer2(h_1))
        u = self.control(x)
        return out, u


N = 499  # sample size
D_in = 2  # input dimension
H1 = 6  # hidden dimension
D_out = 1  # output dimension
torch.manual_seed(10)

lqr = torch.tensor([[-0.8471, -1.6414]])  # lqr solution
x = torch.Tensor(N, D_in).uniform_(-1, 1)
x_0 = torch.zeros([1, 2])
x = torch.cat((x, x_0), 0)


def f_value(x, u):
    v = 6
    l = 1
    y = []

    for r in range(0, len(x)):
        f = [v * torch.sin(x[r][1]),
             v * torch.tan(u[r][0]) / l - (torch.cos(x[r][1]) / (1 - x[r][0]))]
        y.append(f)
    y = torch.tensor(y)
    return y


def dtanh(s):
    # Derivative of activation
    return 1.0 - s ** 2


def Tune(x):
    # Circle function values
    y = []
    for r in range(0, len(x)):
        v = 0
        for j in range(x.shape[1]):
            v += x[r][j] ** 2
        f = [torch.sqrt(v)]
        y.append(f)
    y = torch.tensor(y)
    return y


out_iters = 0
valid = False
while out_iters < 1 and not valid:
    start = timeit.default_timer()
    lqr = torch.tensor([[-23.58639732, -5.31421063]])  # lqr solution
    model = Net(D_in, H1, D_out, lqr)
    L = []
    i = 0
    t = 0
    t = 0
    max_iters = 2000
    learning_rate = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    while i < max_iters and not valid:
        V_candidate, u = model(x)
        X0, u0 = model(x_0)
        f = f_value(x, u)

        Circle_Tuning = Tune(x)
        # Compute lie derivative of V : L_V = ∑∂V/∂xᵢ*fᵢ
        L_V = torch.diagonal(torch.mm(torch.mm(torch.mm(dtanh(V_candidate), model.layer2.weight) \
                                               * dtanh(
            torch.tanh(torch.mm(x, model.layer1.weight.t()) + model.layer1.bias)), model.layer1.weight), f.t()), 0)

        # With tuning
        Lyapunov_risk = (F.relu(-V_candidate) + 2 * F.relu(L_V + 0.8)).mean() \
                        + 1.5 * ((Circle_Tuning - V_candidate).pow(2)).mean() + 1.2 * (X0).pow(2)

        print(i, "Lyapunov Risk=", Lyapunov_risk.item())
        L.append(Lyapunov_risk.item())
        optimizer.zero_grad()
        Lyapunov_risk.backward()
        optimizer.step()
        i += 1

    stop = timeit.default_timer()
    out_iters += 1

    print(stop)
    with torch.no_grad():
        w1 = model.layer1.weight.data.numpy()
        w2 = model.layer2.weight.data.numpy()
        b1 = model.layer1.bias.data.numpy()
        b2 = model.layer2.bias.data.numpy()
        q = model.control.weight.data.numpy()

        X = np.arange(-0.8, 0.8, 0.25, dtype=np.float32)
        X = torch.from_numpy(X)
        Y = np.arange(-0.8, 0.8, 0.25, dtype=np.float32)
        Y = torch.from_numpy(Y)
        V = np.zeros((len(X), len(Y)))

        for indx, x in enumerate(X):
            for indy, y in enumerate(Y):
                xi = torch.tensor([x, y])
                V0, _ = model(xi)
                V[indx][indy] = V0.item()

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(X, Y, V, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()


