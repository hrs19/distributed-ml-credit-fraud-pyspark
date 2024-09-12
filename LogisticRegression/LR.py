import numpy as np
import argparse
from time import time
from operator import add
from pyspark import SparkContext
import warnings
warnings.filterwarnings('ignore')


def readDataRDD(input_file,spark_context):
    """  Read data from an input file and return rdd containing pairs of the form:
                         (x,y)
         where x is a numpy array and y is a real value. The input file should be a 
         'comma separated values' (csv) file: each line of the file should contain x
         followed by y. For example, line:

         1.0,2.1,3.1,4.5

         should be converted to tuple:
        
         (array(1.0,2.1,3.1),4.5,w)
    """ 
    return spark_context.textFile(input_file)\
        	.map(lambda line: line.split(','))\
        	.map(lambda words: (words[:-1],words[-1]))\
        	.map(lambda inp: (np.array([ float(x) for x in inp[0]]), float(inp[1]) if float(inp[1])==1.0 else -1, 289.43800813 if float(inp[1])==1.0 else 0.50086524))



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

def logisticLoss(beta,x,y,w):
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

    
def gradLogisticLoss(beta,x,y,w):
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
    return (-y /(1.0 + np.exp(y * np.dot(beta,x)))*x)

def totalLossRDD(dataRDD,beta,lam = 0.0):
    """  Given a vector beta and a dataset  compute the regularized total logistic loss :
              
               L(β) = Σ_{(x,y) in data}  l(β;x,y)  + λ ||β ||_2^2             
        
         Inputs are:
            - dataRDD: an RDD list containing pairs of the form (x,y), where x is a vector and y is a binary value
            - beta: a vector β
            - lam: the regularization parameter λ
    """
    loss = dataRDD.map(lambda pair: logisticLoss(beta,pair[0],pair[1],pair[2])).reduce(add)
    return (loss + (lam * np.dot(beta,beta)))


def gradTotalLossRDD(dataRDD,beta,lam = 0.0):
    """  Given a vector beta and a dataset perform compute the gradient of regularized total logistic loss :
            
              ∇L(β) = Σ_{(x,y) in data}  ∇l(β;x,y)  + 2λ β   
        
         Inputs are:
            - dataRDD: an RDD containing pairs of the form (x,y), where x is a vector and y is a binary value
            - beta: a vector β
            - linLog: 1 for linear regression, 0 for logistic regression
            - lam: the regularization parameter λ
    """
        
    gradLoss = dataRDD.map(lambda pair: gradLogisticLoss(beta,pair[0],pair[1],pair[2])).reduce(add)
    return (gradLoss + 2.0*lam*beta)


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

def test(dataRDD,beta,Mtest,lam=0.0):
    predictions = dataRDD.map(lambda pair: (np.dot(beta,pair[0]),pair[1])).cache()
    testRMSE = np.sqrt(1*totalLossRDD(dataRDD,beta,lam)/Mtest)
    p = predictions.filter(lambda pair: pair[0]>=0).cache()
    n = predictions.filter(lambda pair: pair[0]<0).cache()
    nump = p.count()
    numn = n.count()
    tp = p.filter(lambda pair: pair[1]==1).count()
    fp = nump-tp
    tn = n.filter(lambda pair: not(pair[1]==1)).count()
    fn = numn-tn  
    print(f'TP = {tp}\tFP = {fp}\tTN = {tn}\tFN = {fn}')
    
    acc = float(tp+tn)/float(nump+numn)
    pre = float(tp)/float(1 if not tp+fp else tp+fp)
    rec = float(tp)/float(1 if not tp+fn else tp+fn)
    
    return(acc,pre,rec,testRMSE)


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
    dataRDD = dataRDD.cache()
    performance = []
    k = 0
    gradNorm = 2*eps
    beta = beta_0
    start = time()
    while k<max_iter and gradNorm > eps:
        obj = totalLossRDD(dataRDD,beta,lam)
        
        grad = gradTotalLossRDD(dataRDD,beta,lam=lam)
        gradNormSq = np.dot(grad,grad)
        gradNorm = np.sqrt(gradNormSq)
        
        fun = lambda x: totalLossRDD(dataRDD,x,lam=lam)
        gamma = lineSearch(fun,beta,grad,obj,gradNormSq)
        
        beta = beta - gamma * grad

        print('k = ',k,'\tt = ',time()-start,'\tL(β_k) = ',obj,'\t||∇L(β_{k-1})||_2 = ',gradNorm,'\tγ = ',gamma)
        
        k = k + 1
        


    return beta, gradNorm, k

'''
        if test_data == None:
            t = time()-start
            print('k = ',k,'\tt = ',t,'\tL(β_k) = ',obj,'\t||∇L(β_k)||_2 = ',gradNorm,'\tgamma = ',gamma)#,'\tACC = ',acc,'\tPRE = ',pre,'\tREC = ',rec)#AUC = ', auc
            # print 'k = ',k,'\tt = ',t,'\tL(β_k) = ',obj,'\t||∇L(β_k)||_2 = ', gradNorm,'\tgamma = ',gamma
        else:
            acc,pre,rec,testRMSE = test(test_data,beta,Mtest,lam)
            t = time()-start
            print('k = ',k,'\tt = ',t,'\tL(β_k) = ',obj,'\t||∇L(β_k)||_2 = ',gradNorm,'\tgamma = ',gamma,'\tACC = ',acc,'\tPRE = ',pre,'\tREC = ',rec)#AUC = ', auc
            # performance.append([testRMSE, gradNorm, k, t, acc, pre, rec, auc])
'''     
        
    # writePerformance(performanceFolder,performance,lam,fold)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Parallel Regression.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--traindata',default=None, help='Input file containing (x,y) pairs, used to train a linear model')
    parser.add_argument('--testdata',default=None, help='Input file containing (x,y) pairs, used to test a linear model')
    parser.add_argument('--beta', default='beta', help='File where beta is stored (when training) and read from (when testing)')
    parser.add_argument('--lam', type=float,default=0.0, help='Regularization parameter λ')
    parser.add_argument('--K', type=float,default=100.00, help='L1 norm threshold')
    parser.add_argument('--max_iter', type=int,default=40, help='Maximum number of iterations')
    parser.add_argument('--eps', type=float, default=0.01, help='ε-tolerance. If the l2_norm gradient is smaller than ε, gradient descent terminates.')
    parser.add_argument('--N',type=int,default=25,help='Level of parallelism')
    # parser.add_argument('--solver',default='GD',choices=['GD', 'FW'],help='GD learns β  via gradient descent, FW learns β using the Frank Wolfe algorithm')
                
    args = parser.parse_args()
    warnings.filterwarnings('ignore')
    sc = SparkContext(appName='Parallel Regression')
    sc.setLogLevel('warn')
    lam = args.lam
    # train_start       
    if args.traindata is not None:
        # Train a linear model β from data, and store it in beta
        print('Reading training data from',args.traindata)
        trainData = readDataRDD(args.traindata,sc)
        trainData = trainData.repartition(args.N).cache()
        
        x,y,w = trainData.take(1)[0]
        dim = len(x)
        beta0 = np.zeros(dim)
        
        print(f"Training sample has {dim} dimensions")
        print('Training on data from', args.traindata, 'with λ =',args.lam,',ε =',args.eps,', max iter = ',args.max_iter)

        beta, gradNorm, i = train(trainData,beta_0=beta0,lam=lam,max_iter=args.max_iter,eps=args.eps)




    if args.testdata is not None:
        # t = time()-start
        # print('k = ',k,'\tt = ',t,'\tL(β_k) = ',obj,'\t||∇L(β_k)||_2 = ',gradNorm,'\tgamma = ',gamma)#,'\tACC = ',acc,'\tPRE = ',pre,'\tREC = ',rec)#AUC = ', auc
        # print 'k = ',k,'\tt = ',t,'\tL(β_k) = ',obj,'\t||∇L(β_k)||_2 = ', gradNorm,'\tgamma = ',gamma
    # else:

        print('Reading test data from',args.testdata)

        testData = readDataRDD(args.testdata,sc)
        testData = testData.repartition(args.N).cache()
        Mtest=testData.count()

        acc,pre,rec,testRMSE = test(testData,beta,Mtest,lam)
        # t = time()-start
        print('ACC = ',acc,'\tPRE = ',pre,'\tREC = ',rec,'\ttestRMSE = ',testRMSE)
        # performance.append([testRMSE, gradNorm, k, t, acc, pre, rec, auc])
