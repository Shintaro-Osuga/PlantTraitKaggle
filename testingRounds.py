from KANlayer import *
from KAN import *
from kan_utils import *

# x = torch.randn(5)
# y = torch.randn(4)
# z = torch.einsum('i,j->ij', x, y)
# print(z.shape)
# model = KANLayer(in_dim=3, out_dim=3, num=5, k=3)
# model = KANLayer(in_dim=3, out_dim=5)
# x = torch.normal(0,1,size=(100,3))
# print(x.shape)
# y = model(x)
# print(y.shape)


# f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
# print(f)
# dataset = create_dataset(f, n_var=2)

def test_premade():
    import matplotlib.pyplot as plt
    import torch.cuda

    # from kan import KAN, create_dataset

    # Let's set the device to be used for the dataset generation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    def xy(x):
        return x[:, [0]] * x[:, [1]]
    
    dataset = create_dataset(
        f=xy,
        n_var=2,
        train_num=10000,
        test_num=1000,
        device=device,
        ranges=[(-10, 10), (-10, 10)]
        )
    
    # plt.scatter(dataset['train_input'][:, [0]].to('cpu'), dataset['train_input'][:, [1]].to('cpu'),
    #         c=dataset['train_label'].flatten().to('cpu'), cmap='viridis')
    # plt.title('Train Dataset')
    # plt.xlabel('x0')
    # plt.ylabel('x1')
    # plt.colorbar()
    # plt.show()
    
    model = KAN(width=[2, 3, 3, 1], grid=100, k=1, device=device)
    # model = nn.Sequential(modelk)
    # print(dataset['train_input'])
    model(dataset['train_input'])
    model.fix_symbolic(0, 0, 0, 'x')
    model.fix_symbolic(0, 1, 0, 'x')
    model.fix_symbolic(0, 0, 1, 'x^2')
    # model.remove_edge(0, 1, 1)
    # model.remove_edge(0, 0, 2)
    model.fix_symbolic(0, 1, 2, 'x^2')
    model.fix_symbolic(1, 0, 0, 'x^2')
    # model.remove_edge(1, 1, 0)
    # model.remove_edge(1, 2, 0)
    # model.remove_edge(1, 0, 1)
    model.fix_symbolic(1, 1, 1, 'x')
    # model.remove_edge(1, 2, 1)
    # model.remove_edge(1, 0, 2)
    # model.remove_edge(1, 1, 2)
    model.fix_symbolic(1, 2, 2, 'x')
    model.fix_symbolic(2, 0, 0, 'x')
    model.fix_symbolic(2, 1, 0, 'x')
    model.fix_symbolic(2, 2, 0, 'x')
    model.plot()
    
    losses = model._train(dataset=dataset, opt='LBFGS', steps=20, device=device)
    
    
    plt.plot(losses['train_loss'], label='train loss')
    plt.plot(losses['test_loss'], label='test loss')
    plt.legend()
    plt.yscale('log')
    plt.show()
    
    
    model.plot()

    model.symbolic_formula()

def test_1():
    h = [[2,4,4],[1,2,43]]
    print(h)
    torch.stack(h)
    print(h.shape)

test_premade()
# test_1()