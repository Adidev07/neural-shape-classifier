"""Train a two-layer neural network with backpropagation to recognize noisy geometric image patterns."""
from kit import *
TITLE='Neural Image Pattern Classifier'
LIVE=False

def dataset(seed,n=600):
    rng=np.random.default_rng(seed);y=rng.integers(0,3,n);images=rng.normal(0,.2,(n,8,8))
    for i,label in enumerate(y):
        if label==0:images[i,:,3:5]+=1
        elif label==1:images[i,3:5,:]+=1
        else:images[i,np.arange(8),np.arange(8)]+=1
    return images.reshape(n,64),y
def init(seed=7):
    rng=np.random.default_rng(seed);return [rng.normal(0,.12,(64,24)),np.zeros(24),rng.normal(0,.12,(24,3)),np.zeros(3)]
def forward(p,x):
    h=np.tanh(x@p[0]+p[1]);z=h@p[2]+p[3];z-=z.max(axis=1,keepdims=True);prob=np.exp(z);prob/=prob.sum(axis=1,keepdims=True);return h,prob
def loss_grad(p,x,y):
    h,prob=forward(p,x);loss=-np.log(np.maximum(prob[np.arange(len(y)),y],1e-15)).mean();dz=prob.copy();dz[np.arange(len(y)),y]-=1;dz/=len(y);dh=(dz@p[2].T)*(1-h*h)
    return float(loss),[x.T@dh,dh.sum(axis=0),h.T@dz,dz.sum(axis=0)]
def train(x,y,epochs=220):
    p=init();history=[]
    for _ in range(epochs):
        loss,g=loss_grad(p,x,y);history.append(loss)
        for v,grad in zip(p,g):v-=.15*grad
    return p,history
def run(d,a):
    x,y=dataset(a.seed);xt,yt=dataset(a.seed+1,200);p,history=train(x,y);pred=forward(p,xt)[1].argmax(axis=1)
    return dict(test_accuracy=float(np.mean(pred==yt)),majority_baseline_accuracy=float(np.mean(yt==np.bincount(y).argmax())),initial_loss=history[0],final_loss=history[-1],classes=['vertical','horizontal','diagonal']),dict(**{f'parameter_{i}':v for i,v in enumerate(p)},loss=np.array(history),test_images=xt[:12].reshape(-1,8,8))

if __name__=="__main__":main(run)
