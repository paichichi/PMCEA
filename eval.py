import torch
import logging
import numpy as np


class evaluateTorch:
    def __init__(self,  device, dev_pair,eval_batch_size=1024, k=10, batch=1024):
        self.device = device
        self.dev_pair = dev_pair
        self.eval_batch_size = eval_batch_size
        self.topk = k
        self.batch = batch

    def sim_results(self, Matrix_A, Matrix_B):
        # A x B.t
        A_sim = torch.mm(Matrix_A, Matrix_B.t())
        return A_sim



    def avg_results(self, Matrix_A):
        k = 10
        avg_results = torch.sum(torch.topk(Matrix_A, k=k)[0], dim=-1) / k
        return avg_results

    def CSLS_results(self, inputs):
        SxT_sim, TxS_avg, SxT_avg,ans_rank = inputs

        TxS_avg, SxT_avg = [torch.unsqueeze(m, dim=1) for m in [TxS_avg, SxT_avg]]
        sim = 2 * SxT_sim - SxT_avg - TxS_avg.t()
        # rank = torch.argsort(sim, dim=-1, descending=True)

        # targets = rank[:, :self.topk]
        # values = torch.gather(sim, 1, targets)

        #     rank = tf.argsort(-sim,axis=-1)
        #     results = tf.where(tf.equal(rank,K.cast(K.tile(ans_rank,[1,len(self.dev_pair)]),dtype="int32")))
        #     return K.expand_dims(results,axis=0)
        # results = torch.where(torch.equal(rank, torch.tensor(ans_rank).unsqueeze(0).repeat(1, len(self.dev_pair)).int()))
        rank = torch.argsort(-sim, dim=-1)
        results = torch.where(torch.eq(rank, torch.tensor(ans_rank).unsqueeze(1).repeat(1, len(self.dev_pair)).to(self.device)))
        return results[1]
        # return torch.unsqueeze(results[1], dim=0)

        # return targets.cpu().numpy(), values.cpu().numpy()

    def CSLS_rank(self, inputs):
        SxT_sim, TxS_avg, SxT_avg,ans_rank = inputs

        TxS_avg, SxT_avg = [torch.unsqueeze(m, dim=1) for m in [TxS_avg, SxT_avg]]
        sim = 2 * SxT_sim - SxT_avg - TxS_avg.t()
        # rank = torch.argsort(sim, dim=-1, descending=True)

        # targets = rank[:, :self.topk]
        # values = torch.gather(sim, 1, targets)

        #     rank = tf.argsort(-sim,axis=-1)
        #     results = tf.where(tf.equal(rank,K.cast(K.tile(ans_rank,[1,len(self.dev_pair)]),dtype="int32")))
        #     return K.expand_dims(results,axis=0)
        # results = torch.where(torch.equal(rank, torch.tensor(ans_rank).unsqueeze(0).repeat(1, len(self.dev_pair)).int()))

        rank = torch.argsort(-sim, dim=-1)
     
        return rank[:,0]

    def CSLS_cal(self, sourceVec, targetVec,m1,m2,evaluate=True):
        #left joint representation
        batch_size = self.eval_batch_size # batch_size = 1024

        SxT_sim = []
        TxS_sim=[]

        for epoch in range(len(sourceVec) // batch_size + 1):
            SxT_sim.append(self.sim_results(sourceVec[epoch * batch_size:(epoch + 1) * batch_size], targetVec))
            TxS_sim.append(self.sim_results(targetVec[epoch * batch_size:(epoch + 1) * batch_size], sourceVec))

        alpha=0.2
        # if evaluate:

        for epoch in range(len(sourceVec) // batch_size + 1):
            SxT_sim[epoch]=SxT_sim[epoch]*(1-alpha)+m1[epoch * batch_size:(epoch + 1) * batch_size,:]*alpha
            TxS_sim[epoch]=TxS_sim[epoch]*(1-alpha)+m2[epoch * batch_size:(epoch + 1) * batch_size,:]*alpha


        SxT_avg = []
        TxS_avg = []
        for epoch in range(len(sourceVec) // batch_size + 1):
            SxT_avg.append(self.avg_results(SxT_sim[epoch]))
            TxS_avg.append(self.avg_results(TxS_sim[epoch]))

        # SxT_sim = self.sim_results(sourceVec, targetVec)
        # SxT_sim = SxT_sim*(1-alpha)+m*alpha

        # SxT_avg = self.avg_results(SxT_sim)

        # TxS_sim = self.sim_results(targetVec, sourceVec)
        # TxS_sim = TxS_sim*(1-alpha)+m*alpha

        # TxS_avg = self.avg_results(TxS_sim)


        if evaluate:
            targets = np.empty((0, self.topk), int)
            targets=[]
            for epoch in range(len(sourceVec) // batch_size + 1):
                ans_rank = np.array([i for i in range(epoch * batch_size,min((epoch+1) * batch_size,len(sourceVec)))])
                temp_targets = self.CSLS_results([SxT_sim[epoch].to(device=self.device),torch.cat((TxS_avg),dim=0).to(device=self.device), SxT_avg[epoch].to(device=self.device),ans_rank])
                # targets = np.concatenate((targets, temp_targets), axis=0)
                targets.append(temp_targets)
                # values = np.concatenate((values, temp_values), axis=0)
            return torch.cat((targets),0)
        else :
            l_rank,r_rank = [],[]
            for epoch in range(len(sourceVec) // batch_size + 1):
                ans_rank = np.array([i for i in range(epoch * batch_size,min((epoch+1) * batch_size,len(sourceVec)))])
                l_rank.append(self.CSLS_rank([SxT_sim[epoch].to(device=self.device), torch.cat((TxS_avg),dim=0).to(device=self.device), SxT_avg[epoch].to(device=self.device),ans_rank]))
                r_rank.append(self.CSLS_rank([TxS_sim[epoch].to(device=self.device), torch.cat((SxT_avg),dim=0).to(device=self.device), TxS_avg[epoch].to(device=self.device),ans_rank]))

            # return np.concatenate(r_rank,axis=1)[0],np.concatenate(l_rank,axis=1)[0] 
            return  torch.cat((r_rank),dim=0).cpu().numpy(),torch.cat((l_rank),dim=0).cpu().numpy()
        
    def test(self, Lvec,Rvec,m1,m2):
        results  = self.CSLS_cal(Lvec,Rvec,m1,m2)
        def cal(results):
            hits1,hits5,hits10,mrr = 0,0,0,0
            for x in results:
                if x < 1:
                    hits1 += 1
                if x < 5:
                    hits5 += 1
                if x < 10:
                    hits10 += 1
                mrr += 1/(x + 1)
            return hits1,hits5,hits10,mrr
        hits1,hits5,hits10,mrr = cal(results)
        print("Hits@1: ",hits1/len(Lvec)," ","Hits@5: ",hits5/len(Lvec)," ","Hits@10: ",hits10/len(Lvec)," ","MRR: ",mrr/len(Lvec))
        return results