import numpy as np

class A:
    def __init__(self):
        self.a = 1
    
    def printt(self):
        print(self.a)
    
    def haha(self):
        print("真的会调用父类的函数")
        self.printt()
    
    
class B(A):
    def __init__(self):
        super(B, self).__init__()
        self.b = 2
    
    def printt(self):
        print(self.b)
        

z = B()
z.printt()
z.haha()