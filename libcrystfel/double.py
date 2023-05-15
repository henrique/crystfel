import torch

#def double(X):
#    return X*2

class Double(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #self.weight = torch.nn.Parameter(torch.rand(N, M))

    def forward(self, input):
        #if input.sum() > 0:
        #  output = self.weight.mv(input)
        #else:
        #  output = self.weight + input
        #return output
        return input * 2

fn = torch.jit.script(Double())
print(fn(torch.ones(3)))

fn.save("double.pt")
