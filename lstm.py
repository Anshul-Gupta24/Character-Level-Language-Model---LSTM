from __future__ import print_function
import numpy as np

f=open('input.txt','r')
inp=f.read()
vocab=list(set(inp))
inp_sz=len(vocab)

char_ind = {ch:i for i,ch in enumerate(vocab)}
ind_char = {i:ch for i,ch in enumerate(vocab)}


#initialize weight matrices
W_f = np.random.randn(100,inp_sz+100)*0.01     #inp_sz+100 X 100 matrix
W_i = np.random.randn(100,inp_sz+100)*0.01     #inp_sz+100 X 100 matrix
W_c = np.random.randn(100,inp_sz+100)*0.01     #inp_sz+100 X 100 matrix
W_o = np.random.randn(100,inp_sz+100)*0.01     #inp_sz+100 X 100 matrix
W_y = np.random.randn(inp_sz,100)*0.01     #100 X inp_sz matrix

b_f = np.zeros((100,1))
b_i = np.zeros((100,1))
b_c = np.zeros((100,1))
b_o = np.zeros((100,1))
b_y = np.zeros((inp_sz,1))

grad_Wf = np.zeros_like(W_f); grad_Wi = np.zeros_like(W_i); grad_Wc = np.zeros_like(W_c); grad_Wo = np.zeros_like(W_o); grad_Wy = np.zeros_like(W_y); grad_bf = np.zeros_like(b_f); grad_bi = np.zeros_like(b_i); grad_bc = np.zeros_like(b_c); grad_bo = np.zeros_like(b_o); grad_by = np.zeros_like(b_y)

h = {}
h[-1] = np.zeros((100,1))
cc = {}
cc[-1] = np.zeros_like(b_o)



def sigmoid(x):
	return 1 / (1 + np.exp(-x))



while(1):
    #over multiple epochs

    #variable initializations
    y = {}
    z = {}
    f = {}
    ix= {}
    cb = {}
    o = {}
    i=0
    c=0

    ya_ind={}

    for t in xrange(len(inp)-1):
        #print("i is " + str(i))
        #print("c is " + str(c))

        #get 1 hot encoding
        ind=char_ind[inp[t]]
        xenc=np.zeros((inp_sz,1))
        xenc[ind]=1


	z[i] = np.concatenate((h[i-1],xenc))
	f[i] = sigmoid(np.dot(W_f,z[i]) + b_f)
	ix[i] = sigmoid(np.dot(W_i,z[i]) + b_i)
	cb[i] = np.tanh(np.dot(W_c,z[i]) + b_c)
	cc[i] = f[i]*cc[i-1] + ix[i]*cb[i]
	o[i] = sigmoid(np.dot(W_o,z[i]) + b_o)

	h[i] = o[i] * np.tanh(cc[i])


        y[i] = np.dot(W_y,h[i]) + b_y
        y[i] = np.exp(y[i]) / np.sum(np.exp(y[i]))

        ya = inp[t+1]
        ya_ind[i] = char_ind[ya]
        

        #backpropagate every 25 chars

        dh_next=np.zeros_like(b_o)
        dC_next=np.zeros_like(b_o)

        d_Wf=np.zeros_like(W_f)
        d_Wc=np.zeros_like(W_c)
        d_Wi=np.zeros_like(W_i)
        d_Wo=np.zeros_like(W_o)
        d_Wy=np.zeros_like(W_y)

        d_bf=np.zeros_like(b_f)
        d_bc=np.zeros_like(b_c)
        d_bi=np.zeros_like(b_i)
        d_bo=np.zeros_like(b_o)
        d_by=np.zeros_like(b_y)

        if(i==24):
            for v in reversed(xrange(25)):
                dy = np.copy(y[v])
                dy[ya_ind[v]]-=1
                
                d_Wy += np.dot(dy, h[v].T)
                d_by += dy
                
                dh = np.dot(W_y.T, dy) + dh_next

                do = dh * np.tanh(cc[v])
                d_oo = do*o[v]*(1 - o[v])
		d_Wo += np.dot(d_oo, z[v].T)
		d_bo += d_oo

                dC = dh * o[v] * (1 - np.tanh(cc[v]) * np.tanh(cc[v])) + dC_next
                
		dCb = dC * ix[v]
		dCCb = dCb * (1 - cb[v] * cb[v])
		d_Wc += np.dot(dCCb, z[v].T)
		d_bc += dCCb

                di = dC * cb[v]
		dii = di * ix[v] * (1 - ix[v])
		d_Wi += np.dot(dii, z[v].T)
		d_bi += dii

                df = dC * cc[v-1]
		dff = df * f[v] * (1 - f[v])
		d_Wf += np.dot(dff, z[v].T)
		d_bf += dff
			
		dz = np.dot(W_f.T, dff) \
        		+ np.dot(W_i.T, dii) \
        		+ np.dot(W_c.T, dCCb) \
        		+ np.dot(W_o.T, d_oo)
    		
		dh_prev = dz[:100, :]
                dC_next = dC * f[v]

            #clip gradients to prevent exploding gradients problem
            for dparam in [d_Wf, d_bf, d_Wi, d_bi, d_Wc, d_bc, d_Wo, d_bo, d_Wy, d_by]:
                np.clip(dparam, -5, 5, out=dparam) 

            i=-1
            c+=1

	    h[-1]=np.copy(h[24])
	    cc[-1]=np.copy(cc[24])


            #update weights with AdaGrad
            for weights,grad,old_grad in zip([W_f, W_i, W_c, W_o, W_y, b_f, b_i, b_c, b_o, b_y],[d_Wf, d_Wi, d_Wc, d_Wo, d_Wy, d_bf, d_bi, d_bc, d_bo, d_by],[grad_Wf, grad_Wi, grad_Wc, grad_Wo, grad_Wy, grad_bf, grad_bi, grad_bc, grad_bo, grad_by]):
                old_grad+=grad*grad
                weights += -(1e-1) * grad / np.sqrt(old_grad + 1e-8)



        #sample every 100 iterations
        if(c==100):
            

            #pick 1st character at random
            #ip = np.zeros((inp_sz,1))
            #ip_1 = np.random.randint(0,inp_sz)
            #ip[ip_1]=1
            ip = z[0][100:]


            #print a sample of size n

            n=200
            h0 = np.copy(h[-1])
            ccs = np.copy(cc[-1])

            for b in range(n):

		zs = np.concatenate((h0,ip))
		fs = sigmoid(np.dot(W_f,zs) + b_f)
		i_s = sigmoid(np.dot(W_i,zs) + b_i)
		cs = np.tanh(np.dot(W_c,zs) + b_c)
		ccs = fs*ccs + i_s*cs
		os = sigmoid(np.dot(W_o,zs) + b_o)

		h0 = os * np.tanh(ccs)


        	ys = np.dot(W_y,h0) + b_y
        	ys = np.exp(ys) / np.sum(np.exp(ys))


                op_ind = int(np.random.choice(range(inp_sz), p=ys.ravel()))   #sample character according to output probabilties
		
            	ip = np.zeros((inp_sz,1))
		ip[op_ind]=1
                ch=ind_char[op_ind]
                #h0=np.copy(h1)
                print(ch,end='')
            
            #print ("at iteration " + string(t) + ", loss is: " + string(loss))  #print loss
            print ('')
            print ('')
            print ('')
            print ('')
            print ('')
            
            c=0

        i+=1
