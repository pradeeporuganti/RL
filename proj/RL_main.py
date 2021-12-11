from functions import *
import torch
import torch.nn as nn
import torch.nn.functional as Fu

# class Net(torch.nn.Module):

#    def __init__(self, n_input, n_output):
#        super(Net, self).__init__()

#        self.control1 = torch.nn.Linear(n_input, 64)
#        self.control2 = torch.nn.Linear(64, n_output)

#    def forward(self, x):
#        sigmoid = nn.Tanh()
#        h_1 = sigmoid(self.control1(x))
#        u = self.control2(h_1)
#        return u

class NeuralNet(nn.Module):

    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(NeuralNet, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation):
        state = torch.tensor(observation, dtype=torch.float).to(self.device)
        x = Fu.relu(self.fc1(state))
        x = Fu.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class Agent(object):

    def __init__(self, alpha, beta, input_dims, gamma=0.99, n_actions=2,
                 layer1_size=256, layer2_size=256, n_outputs=1):
        self.gamma = gamma
        self.log_probs = None
        self.n_outputs = n_outputs
        self.actor = NeuralNet(alpha, input_dims, layer1_size,
                               layer2_size, n_actions=n_actions)
        self.critic = NeuralNet(beta, input_dims, layer1_size,
                                layer2_size, n_actions=1)

    def choose_action(self, observation):
        #print(observation)
        #print(self.actor.forward(observation))
        mu, sigma = self.actor.forward(observation)[0]
        sigma = torch.exp(sigma)
        action_probs = torch.distributions.Normal(mu, sigma)
        probs = action_probs.sample(sample_shape=torch.Size([self.n_outputs]))
        self.log_probs = action_probs.log_prob(probs).to(self.actor.device)
        action = 1.5*torch.tanh(probs)


        return action.item()

    def learn(self, state, reward, new_state, done):
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        critic_value_ = self.critic.forward(new_state)
        critic_value = self.critic.forward(state)

        reward = torch.tensor(reward, dtype=torch.float).to(self.actor.device)
        delta = reward + self.gamma*critic_value_*(1-int(done)) - critic_value

        actor_loss = -self.log_probs * delta
        critic_loss = delta**2

        (actor_loss + critic_loss).backward()

        self.actor.optimizer.step()
        self.critic.optimizer.step()

#def train(u_cl, xx):
    # get data
    #X = torch.from_numpy(xx.astype(np.float32))
    #y = torch.from_numpy(u_cl.astype(np.float32))

    #n_samples, n_features = X.shape
    #_, n_out = y.shape

    #input_size = n_features
    #output_size = n_out

    #nn_cont = Net(input_size, output_size)

    # Loss criterion
    #criterion = nn.MSELoss()
    #learning_rate = 0.01
    #optimizer = torch.optim.Adam(nn_cont.parameters(), lr=learning_rate)

    #num_epochs = 1000
    #for epoch in range(num_epochs):
        # Forward pass and loss
        #y_predicted = nn_cont(X)
        #loss = criterion(y_predicted, y)

        # Backward pass and update
        #loss.backward()
        #optimizer.step()

        # zero grad before new step
        #optimizer.zero_grad()

        #if (epoch + 1) % 10 == 0:
        #    print(f'epoch: {epoch + 1}, loss = {loss.item():.4f}')

    #with torch.no_grad():
        #nn_xx = np.array([20, 0, 0, 0, 0, 0], dtype=np.float32)
        #nn_xx = np.reshape(nn_xx, (1, 6))

        #for i in range(45):
            #nn_u_apply = nn_cont(torch.from_numpy(nn_xx[i, :]))
            #nn_xx_next = np.array(step(nn_xx[i, :], nn_u_apply.detach().numpy()), dtype=np.float32)
            #nn_xx = np.concatenate((nn_xx, np.reshape(nn_xx_next, (1, 6))), axis=0)

    #plots(nn_xx, 'r-')
