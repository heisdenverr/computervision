from typing import Any
from main import Vipy
class Engine:

    def __init__(self: Vipy.toarray, data: Vipy.toarray,  children=(), _ops=" "):
        self.data = data
        self._prev = set(children)
        self._ops = _ops
        self.grad = 0
        self._backward = lambda: None


    def __repr__(self):
        return f"degrad (data={self.data})"

    def __call__(self):
        return Engine(self)
    
    

    def __add__(self, other):
        other = other if isinstance(other, Engine) else Engine(other)
        out = Engine(Vipy.toarray((self.data + other.data, (self, other), '+')))

        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

            
        
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Engine) else Engine(other)
        out = Engine(Vipy.toarray(self.data * other.data, (self, other), '*'))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def backward(self):

        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)


        self.grad = 1
        for k in reversed(topo):
            k._backward()
    
        

    
    
n = 3
v = 4
a = [[0, 1],
      [1, 0]]

b = [[3, 4],
      [1, 2]]
engine_grad = Engine(a + b)
print((engine_grad + b).backward)
print(Engine(a).backward)
print(engine_grad.grad)