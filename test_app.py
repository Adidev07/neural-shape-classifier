import unittest
from app import *
class DomainTests(unittest.TestCase):
    def test_gradient(self):
        x,y=dataset(1,6);p=init();_,g=loss_grad(p,x,y);old=p[0][0,0];eps=1e-5;p[0][0,0]=old+eps;plus=loss_grad(p,x,y)[0];p[0][0,0]=old-eps;minus=loss_grad(p,x,y)[0];self.assertAlmostEqual((plus-minus)/(2*eps),g[0][0,0],places=6)
    def test_learning(self):
        x,y=dataset(1,200);p,h=train(x,y,80);self.assertLess(h[-1],h[0]*.3)
    def test_softmax_sum(self):np.testing.assert_allclose(forward(init(),np.ones((3,64)))[1].sum(axis=1),1)
