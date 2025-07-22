import torch
import matplotlib.pyplot as plt
from torch import nn, optim

class Net(nn.Module):
    def __init__(self, input_features, num_hidden, output_features):
        super(Net, self).__init__()
        self.hidden = nn.Linear(input_features, num_hidden)
        self.output = nn.Linear(num_hidden, output_features)

    def forward(self, x):
        x = nn.functional.relu(self.hidden(x))
        x = self.output(x)

        return x

def draw(x, y, output, loss):
    plt.cla()
    plt.scatter(x.numpy(), y.numpy())
    plt.plot(x.numpy(), output.data.numpy(), 'r-', lw=5)
    plt.text(min(x.numpy())+0.5, min(y.numpy()), f'Loss: {loss.item()}')
    plt.pause(0.005)

def train(model, criterion, optimizer, x, y, num_epochs):
    for epoch in range(num_epochs):
        output = model(x)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 40 == 0:
            draw(x, y, output, loss)
    return output, loss

def main():
    x = torch.unsqueeze(torch.linspace(-3, 3, 1000), dim=1)
    y = torch.sin(x) + 0.3*torch.randn(x.size())

    net = Net(input_features=1, num_hidden=50, output_features=1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.05)

    input = x
    target = y

    output, loss = train(net, criterion, optimizer, input, target, num_epochs=10000)

    print(f'loss={loss.item()}')
    print(f'output weight:{net.output.weight}')
    plt.show()

if __name__ == "__main__":
    main()