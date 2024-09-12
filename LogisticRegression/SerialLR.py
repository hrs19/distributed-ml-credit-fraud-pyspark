import numpy as np
import argparse
from time import time
from operator import add
# from pyspark import SparkContext
import warnings
warnings.filterwarnings('ignore')

def readData(input_file):
    """  Read data from an input file. Each line of the file contains tuples of the form

                    (x,y)  

         x is a dictionary of the form:                 

           { "feature1": value, "feature2":value, ...}

         and y is a binary value +1 or -1.

         The result is stored in a list containing tuples of the form
                 (SparseVector(x),y)             

    """ 
    listSoFar = []
    with open(input_file,'r') as fh:
        for line in fh:
            # for items in line:
            word = line.split(',')
            # print(word)
            # print(word[:-1],word[-1])
            # print((word[:-1]),eval(word[-1]))
            x = []
            for items in word[:-1]:
                # print(items)
                # print(eval(items))

                x.append(eval(items))

            (x,y) = x,eval(word[-1])

            # (x,y) = eval(line)

            # x = SparseVector(x)
            listSoFar.append((x,y))

    # print(listSoFar[:2])
    return listSoFar

def readBeta(input):
    """ Read a vector β from CSV file input
    """
    with open(input,'r') as fh:
        str_list = fh.read().strip().split(',')
        return np.array( [float(val) for val in str_list] )           

def writeBeta(output,beta):
    """ Write a vector β to a CSV file ouptut
    """
    with open(output,'w') as fh:
        fh.write(','.join(map(str, beta.tolist()))+'\n')

def logisticLoss(beta,x,y):
    """
        Given beta as an array, x as an array, and a binary value y in {-1,+1}, compute the logistic loss
               
                l(β;x,y) = log( 1.0 + exp(-y * <β,x>) )
                l(β;x,y) = w*log( 1.0 + exp(-y * <β,x>) )

                
	The input is:
	    - beta: a vector β
	    - x: a vector x
        - y: a binary value in {-1,+1}

    """
    #return (w*np.log(1.0 + np.exp(np.float128(-1.0*y * np.dot(beta,x)))))
    return (np.log(1.0 + np.exp(np.float128(-1.0*y * np.dot(beta,x)))))

def gradLogisticLoss(beta,x,y):
    """
        Given beta as an array, x as an array, and
        a binary value y in {-1,+1}, compute the gradient of the logistic loss 

              ∇l(B;x,y) = -y / (1.0 + exp(y <β,x> )) * x
              ∇l(B;x,y) = -w*y / (1.0 + exp(y <β,x> )) * x

	The input is:
	    - beta: a vector β
	    - x: a vector x
        - y: a binary value in {-1,+1}


    """
    # return (-w*y /(1.0 + np.exp(y * np.dot(beta,x)))*x)
    # print('type y: ',type(y))
    # print('type x: ',type(x[0]))
    # print('type beta: ',beta)
    # print(np.exp(y * np.dot(beta,x)))
    # print('2*x:',2.2*x)
    return -y /(1.0 + np.exp(y * np.dot(beta,x))) * np.asarray(x)


def totalLoss(data,beta,lam = 0.0):
    """  Given a sparse vector beta and a dataset  compute the regularized total logistic loss :
              
               L(β) = Σ_{(x,y) in data}  l(β;x,y)  + λ ||β ||_2^2             
        
         Inputs are:
            - data: a python list containing pairs of the form (x,y), where x is a sparse vector and y is a binary value
            - beta: a sparse vector β
            - lam: the regularization parameter λ

         Output is:
            - The loss L(β) 
    """
    x,y = data[0]
    L = logisticLoss(beta,x,y)
    for x,y in data[1:]:
        L +=  logisticLoss(beta,x,y)

    # print(beta,L)
    return L+(lam*(beta.dot(beta)))

    # return L+(lam*(np.linalg.norm(beta)**2))

def gradTotalLoss(data,beta, lam = 0.0):
    """  Given a sparse vector beta and a dataset perform compute the gradient of regularized total logistic loss :
            
              ∇L(β) = Σ_{(x,y) in data}  ∇l(β;x,y)  + 2λ β   
        
         Inputs are:
            - data: a python list containing pairs of the form (x,y), where x is a sparse vector and y is a binary value
            - beta: a sparse vector β
            - lam: the regularization parameter λ

         Output is:
            - The gradient ∇L(β) 
    """    
    # del_L = np.zeros(len(beta))
    x,y = data[0]
    del_L = gradLogisticLoss(beta,x,y)
    for x,y in data[1:]:
        # print(gradLogisticLoss(beta,x,y))
        # print(logisticLoss(beta,x,y))
        del_L +=  gradLogisticLoss(beta,x,y)
    # print(beta)
    # print(del_L+(2*lam*beta))
    return del_L+(2*lam*beta)

def lineSearch(fun,x,grad,fx,gradNormSq, a=0.2,b=0.6):
    """ Given function fun, a current argument x, and gradient grad=∇fun(x), 
        perform backtracking line search to find the next point to move to.
        (see Boyd and Vandenberghe, page 464).

	
        Inputs are:
	    - fun: the objective function f.
	    - x: the present input (a Vector)
        - grad: the present gradient (as Vector)
        - fx: precomputed f(x) 
        - grad: precomputed ∇f(x)
        - Optional parameters a,b  are the parameters of the line search.

        Given function fun, and current argument x, and gradient grad=∇fun(x), the function finds a t such that
        fun(x - t * ∇f(x)) <= f(x) - a * t * <∇f(x),∇f(x)>

        The return value is the resulting value of t.
    """
    t = 1.0
    while fun(x-t*grad) > fx- a * t * gradNormSq:
        t = b * t
    return t    

def train(dataRDD,beta_0,lam,max_iter,eps):
    """
    Perform logistic regression:
 

	where
             - dataRDD: an rdd containing pairs of the form (x,y)
             - beta_0: the starting vector β
             - lam:  is the regularization parameter λ
             - max_iter: maximum number of iterations of gradient descent
             - eps: upper bound on the l2 norm of the gradient
             - performanceFolder: the save location of the performance statistics
             - fold: the number of the current fold
             - linLog: 1 for linear regression 0 for logistic regression
             - Mtest: number of test samples
             - test_data: the validation data to be used
 

	The function returns:
	     -beta: the trained β, 
	     -gradNorm: the norm of the gradient at the trained β, and
             -k: the number of iterations performed
    """
    # dataRDD = dataRDD.cache()
    performance = []
    k = 0
    gradNorm = 2*eps
    beta = beta_0
    start = time()
    while k<max_iter and gradNorm > eps:
        obj = totalLoss(dataRDD,beta,lam)
        
        grad = gradTotalLoss(dataRDD,beta,lam=lam)
        gradNormSq = np.dot(grad,grad)
        gradNorm = np.sqrt(gradNormSq)
        
        fun = lambda x: totalLoss(dataRDD,x,lam=lam)
        gamma = lineSearch(fun,beta,grad,obj,gradNormSq)
        
        beta = beta - gamma * grad

        print('k = ',k,'\tt = ',time()-start,'\tL(β_k) = ',obj,'\t||∇L(β_{k-1})||_2 = ',gradNorm,'\tγ = ',gamma)
        
        k = k + 1
        


    return beta, gradNorm, k

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Logistic Regression.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.add_argument('traindata',default=None, help='Input file containing (x,y) pairs, used to train a logistic model')
    parser.add_argument('--testdata',default=None, help='Input file containing (x,y) pairs, used to test a logistic model')
    parser.add_argument('--beta', default='beta', help='File where beta is stored (when training) and read from (when testing)')
    parser.add_argument('--lam', type=float,default=0.0, help='Regularization parameter λ')
    parser.add_argument('--max_iter', type=int,default=10, help='Maximum number of iterations')
    parser.add_argument('--eps', type=float, default=0.1, help='ε-tolerance. If the l2_norm of the gradient is smaller than ε, gradient descent terminates.') 
    parser.add_argument('--N',type=int,default=25,help='Level of parallelism')

    test_group = parser.add_mutually_exclusive_group(required=False)
    test_group.add_argument('--online_test', dest='test_while_training', action='store_true',help="Test during training. --testdata must be provided.")
    test_group.add_argument('--end_test_only', dest='test_while_training', action='store_false',help="Suppress testing during training. If --testdata is provided, testing will happen only at the very end of the training.")
    parser.set_defaults(test_while_training=False)
 
    args = parser.parse_args()
    

    print('Reading training data from',args.traindata)
    readtime = time()

    traindata = readData(args.traindata)
    print('readdata time:',time()-readtime)

    print('Read',len(traindata),'data points with')
    
    if args.testdata:
        print('Reading test data from',args.testdata)
        readtime = time()
        testdata = readData(args.testdata)
        print('readdata time:',readtime-time())

        print('Read',len(testdata),'data points with',len(testdata[0])-1,'features in total.')
    else:
        testdata = None

    print(traindata[0][0])
    print(traindata[0])

    x,y = traindata[0]
    dim = len(x)
    beta0 = np.zeros(dim)

    beta, gradNorm, k = train(traindata,beta_0=beta0,lam=args.lam,max_iter=args.max_iter,eps=args.eps) 