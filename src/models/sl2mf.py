# import pdb
# import time
import numpy as np
# import wandb

np.random.seed(456)

class LMF:
    def __init__(self, num_factors=10, nn_size=100, theta=1.0, reg=0.01, alpha=0.01, beta=0.01, beta1=0.01, beta2=0.01,
                 max_iter=30, seed=123):
        self.num_factors = num_factors
        self.nn_size = nn_size
        self.theta = theta
        self.reg = reg
        self.alpha = alpha
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2

        self.max_iter = max_iter
        self.seed = seed

    def deriv(self):
        vec_deriv = -np.dot(self.weight_IntMat, self.U)
        A = np.dot(self.U, self.U.T)
        A = np.exp(A)
        A /= (A + self.ones)
        A = self.W * A
        vec_deriv += np.dot(A, self.U)
        vec_deriv += self.reg * self.U
        if self.alpha > 0 and self.GoSim is not None:
            vec_deriv += self.alpha * np.dot(self.GoLap, self.U)
        if self.beta > 0 and self.GoCCSim is not None:
            vec_deriv += self.beta * np.dot(self.CCLap, self.U)
        if self.beta1 > 0 and self.PPISim is not None:
            vec_deriv += self.beta1 * np.dot(self.PPILap, self.U)
        # if self.beta2 > 0 and self.COPATHSim is not None:
        #    vec_deriv += self.beta2*np.dot(self.PATHLap, self.U)
        return vec_deriv

    def deriv_long(self):
        # vec_deriv = -np.dot(self.weight_IntMat, self.U)
        P = np.dot(self.U, self.U.T)
        P = np.exp(P)
        P /= (P + self.ones)
        P = P * self.mask
        self.IntMat = self.IntMat * self.mask
        P_M = P - self.IntMat
        P_M = P_M * self.W
        vec_deriv += np.dot(P_M, self.U)
    
        vec_deriv += self.reg * self.U
        if self.alpha > 0 and self.GoSim is not None:
            vec_deriv += self.alpha * np.dot(self.GoLap, self.U)
        if self.beta > 0 and self.GoCCSim is not None:
            vec_deriv += self.beta * np.dot(self.CCLap, self.U)
        if self.beta1 > 0 and self.PPISim is not None:
            vec_deriv += self.beta1 * np.dot(self.PPILap, self.U)
        # if self.beta2 > 0 and self.COPATHSim is not None:
        #    vec_deriv += self.beta2*np.dot(self.PATHLap, self.U)
        return vec_deriv

    def compute_loss(self):
        A = np.dot(self.U, self.U.T)
        B = A * self.weight_IntMat
        B = B * self.mask
        loss = -np.sum(B)
        A = np.exp(A)
        A += self.ones
        A = np.log(A)
        A = self.W * A
        A = A * self.mask
        loss += np.sum(A)
        loss = 0.5 * loss + 0.5 * self.reg * np.sum(np.square(self.U))
        if self.alpha > 0 and self.GoSim is not None:
            loss += 0.5 * self.alpha * np.sum(np.diag((np.dot(self.U.T, self.GoLap)).dot(self.U)))
        if self.beta > 0 and self.GoCCSim is not None:
            loss += 0.5 * self.beta * np.sum(np.diag((np.dot(self.U.T, self.CCLap)).dot(self.U)))
        if self.beta1 > 0 and self.PPISim is not None:
            loss += 0.5 * self.beta1 * np.sum(np.diag((np.dot(self.U.T, self.PPILap)).dot(self.U)))
            # if self.beta2 > 0 and self.COPATHSim is not None:
        #    loss += 0.5*self.beta2*np.sum(np.diag((np.dot(self.U.T, self.PATHLap)).dot(self.U)))
        return loss

    def build_KNN_matrix(self, S, nn_size):
        m, n = S.shape
        X = np.zeros((m, n))
        for i in range(m):
            ii = np.argsort(S[i, :])[::-1][:min(nn_size, n)]
            X[i, ii] = S[i, ii]
        return X

    def compute_laplacian_matrix(self, S, nn_size):
        if nn_size > 0:
            S1 = self.build_KNN_matrix(S, nn_size)
            x = np.sum(S1, axis=1)
            return np.diag(x) - S1
        else:
            x = np.sum(S, axis=1)
            return np.diag(x) - S

    def fix(self, IntMat, W, mask, GoSim=None, GoCCSim=None, PPISim=None, COPATHSim=None, run=None):
        '''
        IntMat: The sparse interaction matrix
        W: the weighting matrix
        GoSim: the GO similarity matrix
        TpSim: the topology structure similarity matrix
        '''
        self.IntMat, self.W, self.mask = IntMat, W, mask
        self.GoSim, self.GoCCSim, self.PPISim, self.COPATHSim = GoSim, GoCCSim, PPISim, COPATHSim
        self.weight_IntMat = self.IntMat * self.W
        if self.alpha > 0 and self.GoSim is not None:
            self.GoLap = self.compute_laplacian_matrix(self.GoSim, self.nn_size)
        if self.beta > 0 and self.GoCCSim is not None:
            self.CCLap = self.compute_laplacian_matrix(self.GoCCSim, self.nn_size)
        if self.beta1 > 0 and self.PPISim is not None:
            self.PPILap = self.compute_laplacian_matrix(self.PPISim, self.nn_size)
        # if self.beta2 > 0 and self.COPATHSim is not None:
        #    self.PATHLap = self.compute_laplacian_matrix(self.COPATHSim, self.nn_size)

        self.num_rows = IntMat.shape[0]
        self.ones = np.ones((self.num_rows, self.num_rows))
        prng = np.random.RandomState(self.seed)
        self.U = np.sqrt(1 / float(self.num_factors)) * prng.normal(size=(self.num_rows, self.num_factors))
        grad_sum = np.zeros((self.num_rows, self.num_factors))
        last_log = self.compute_loss()
        for t in range(self.max_iter):
            print("iteration: %d" % t)
            grad = self.deriv()
            grad_sum += np.square(grad)
            vec_step_size = self.theta / np.sqrt(grad_sum)
            self.U -= vec_step_size * grad
            curr_log = self.compute_loss()
            delta_log = (curr_log - last_log) / abs(last_log)
            run.log({
                'train_loss':last_log,
                'delta_log':delta_log
            })
            # print "iter:%s, curr_loss:%s, last_loss:%s, delta_loss:%s" % (t, curr_log, last_log, delta_log)
            if abs(delta_log) < 1e-5:
                break
            last_log = curr_log
        # print "complete model training"

    def smooth_prediction(self, test_data):
        pass

    def predict(self, test_data):
        val = np.sum(self.U[test_data[:, 0], :] * self.U[test_data[:, 1], :], axis=1)
        val = np.exp(val)
        val = val / (1 + val)
        return val

    def __str__(self):
        return "Model: LMF, num_factors:%s, nn_size:%s, theta:%s, reg:%s, alpha:%s, beta:%s, max_iter:%s, seed:%s" % (
        self.num_factors, self.nn_size, self.theta, self.reg, self.alpha, self.beta, self.max_iter, self.seed)


class NRLMF(LMF):

    def __init__(self, d=10, c=5, K=5, N=5, theta=1.0, reg_d=0.6, reg_t=0.6, alpha=0.1, beta=0.1, max_iter=100):
        LMF.__init__(self, d, c, N, theta, reg_d, reg_t, max_iter)
        self.K = K
        self.alpha = alpha
        self.beta = beta

    def deriv(self, drug):
        if drug:
            vec_deriv = np.dot(self.intMat, self.V)
        else:
            vec_deriv = np.dot(self.intMat.T, self.U)
        A = np.dot(self.U, self.V.T)
        A = np.exp(A)
        A /= (A + self.ones)
        A = self.intMat1 * A
        if drug:
            vec_deriv -= np.dot(A, self.V)
            if self.K > 0 and self.alpha > 0:
                vec_deriv -= self.reg_d*self.U+self.alpha*np.dot(self.DL, self.U)
            else:
                vec_deriv -= self.reg_d*self.U
        else:
            vec_deriv -= np.dot(A.T, self.U)
            if self.K > 0 and self.beta > 0:
                vec_deriv -= self.reg_t*self.V+self.beta*np.dot(self.TL, self.V)
            else:
                vec_deriv -= self.reg_t*self.V
        return vec_deriv

    def log_likelihood(self):
        loglik = 0
        A = np.dot(self.U, self.V.T)
        B = A * self.intMat
        loglik += np.sum(B)
        A = np.exp(A)
        A += self.ones
        A = np.log(A)
        A = self.intMat1 * A
        loglik -= np.sum(A)
        loglik -= 0.5 * self.reg_d * np.sum(np.square(self.U))+0.5 * self.reg_t * np.sum(np.square(self.V))
        if self.K > 0 and self.alpha > 0:
            loglik -= 0.5 * self.alpha * np.sum(np.diag((np.dot(self.U.T, self.DL)).dot(self.U)))
        if self.K > 0 and self.beta > 0:
            loglik -= 0.5 * self.beta * np.sum(np.diag((np.dot(self.V.T, self.TL)).dot(self.V)))
        return loglik

    def construct_neighborhood(self, drugMat, targetMat):
        self.dsMat = drugMat - np.diag(np.diag(drugMat))
        self.tsMat = targetMat - np.diag(np.diag(targetMat))
        if self.K > 0:
            S1 = self.get_nearest_neighbors(self.dsMat, self.K)
            self.DL = self.laplacian_matrix(S1)
            S2 = self.get_nearest_neighbors(self.tsMat, self.K)
            self.TL = self.laplacian_matrix(S2)
        else:
            self.DL = self.laplacian_matrix(self.dsMat)
            self.TL = self.laplacian_matrix(self.tsMat)

    def laplacian_matrix(self, S):
        x = np.sum(S, axis=0)
        y = np.sum(S, axis=1)
        L = 0.5*(np.diag(x+y) - (S+S.T))  # laplacian
        return L

    def get_nearest_neighbors(self, S, size=5):
        m, n = S.shape
        X = np.zeros((m, n))
        for i in xrange(m):
            ii = np.argsort(S[i, :])[::-1][:min(size, n)]
            X[i, ii] = S[i, ii]
        return X

    def fix_model(self, W, intMat, drugMat, targetMat, seed=None):
        self.num_drugs, self.num_targets = intMat.shape
        self.ones = np.ones((self.num_drugs, self.num_targets))
        self.intMat = self.c*intMat*W
        self.intMat1 = (self.c-1)*intMat*W + self.ones
        x, y = np.where(self.intMat > 0)
        self.train_drugs, self.train_targets = set(x.tolist()), set(y.tolist())
        self.construct_neighborhood(drugMat, targetMat)
        self.AGD_optimization(seed)

    def __str__(self):
        return "Model: NRLMF, d:%s, c:%s, K:%s, N:%s, theta:%s, reg_d:%s, reg_t:%s, alpha:%s, beta:%s, max_iter:%s" % (self.d, self.c, self.K, self.N, self.theta, self.reg_d, self.reg_t, self.alpha, self.beta, self.max_iter)


# if __name__ == "__main__":
#     from functions import *
#     cv_setting = 1
#     dataset = "nr"
#     intMat, drugMat, targetMat = load_data_from_file(dataset, "../dataset/")
#     if cv_setting == 1:  # CV setting S1
#         X, D, T, cv = intMat, drugMat, targetMat, 1
#     if cv_setting == 2:  # CV setting S2
#         X, D, T, cv = intMat, drugMat, targetMat, 0
#     if cv_setting == 3:  # CV setting S3
#         X, D, T, cv = intMat.T, targetMat, drugMat, 0
#
#     max_aupr, aupr_opt = 0, []
#     max_auc, auc_opt = 0, []
#     seeds = [7771, 8367, 22, 1812, 4659]
#     # seeds = [7771, 8367, 22, 1812, 4659, 1809, 1211, 5841, 6392, 6005, 481, 2458, 5191, 246, 3847, 8212, 5336, 3058, 3701, 1247]
#     # seeds = np.random.choice(10000, 5, replace=False)
#     cv_data = cross_validation(X, seeds, cv)
#
#     tic = time.clock()
#     model = NRLMF(d=100, c=5, K=5, N=5, theta=2**(-1), reg_d=2**(-3), reg_t=2**(-3), alpha=2**(-2), beta=2**(-3), max_iter=100)
#     cmd = str(model)
#     print "dataset:"+dataset+" cvs:"+str(cv_setting)+"\n"+cmd
#     aupr_vec, auc_vec = train(model, cv_data, X, D, T)
#     aupr_avg, aupr_st = mean_confidence_interval(aupr_vec)
#     auc_avg, auc_st = mean_confidence_interval(auc_vec)
#     print "AUPR: %s, AUC:%s, AUPRst:%s, AUCst:%s, Time:%s" % (aupr_avg, auc_avg, aupr_st, auc_st, time.clock() - tic)
    # if aupr_avg > max_aupr:
    #     max_aupr = aupr_avg
    #     aupr_opt = [cmd, aupr_avg, auc_avg]
    # if auc_avg > max_auc:
    #     max_auc = auc_avg
    #     auc_opt = [cmd, aupr_avg, auc_avg]
    # cmd = "Optimal Parameters for AUPR optimization:\n%s\n" % aupr_opt[0]
    # cmd += "AUPR: %s, AUC: %s\n" % (aupr_opt[1], aupr_opt[2])
    # cmd += "Optimal Parameters for AUC optimization:\n%s\n" % auc_opt[0]
    # cmd += "AUPR: %s, AUC: %s" % (auc_opt[1], auc_opt[2])
    # print "\n"+cmd
    # with open("../output/nrlmf_results.txt", "a+") as outf:
    #     outf.write("Dataset:"+dataset+"\n"+cmd+"\n\n")
    # write_metric_vector_to_file(aupr_vec, "../output/nrlmf_aupr_"+str(cv_setting)+"_"+dataset+".txt")
    # write_metric_vector_to_file(auc_vec, "../output/nrlmf_auc_"+str(cv_setting)+"_"+dataset+".txt")
