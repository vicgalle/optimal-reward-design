from parallax import Module, Parameter, ParamInit
import jax

class SGD(Module):
    lr : Parameter
    x0 : Parameter

    def __init__(self):
        super().__init__()
        self.lr = 0.1
        self.x0 = 0.5

    def forward(self, input):
        x = self.x0
        T, loss = input
        g_loss = jax.grad(loss)
        for _ in range(T):
            x = x - self.lr * g_loss(x)
        return x

sgd = SGD()
print(sgd)

T = 2
def inner_loss(x):
    return jax.numpy.sum(x ** 2)

for i in range(10):

    # Jax style grad compute -> tree-shaped immutable
    l = inner_loss(sgd((T, inner_loss)))
    print(l, sgd.lr, sgd.x0)

    g = sgd.grad((T, inner_loss))
    
    # Grad Update -> tree-shaped
    sgd = jax.tree_util.tree_multimap(lambda p, g: p - 0.05 * g, sgd, g)
    