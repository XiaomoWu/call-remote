# credits to https://github.com/AvivNavon/AuxiLearn/blob/master/experiments/weight_methods.py

import torch
import torch.nn.functional as F

from abc import abstractmethod
from omegaconf import ListConfig


class WeightingMethod:
    @abstractmethod
    def backward(self, losses, *args, **kwargs):
        pass


class Fixed(WeightingMethod):
    def __init__(self, init_task_weights, **kwargs):
        self.task_weights = torch.tensor(init_task_weights)

    def backward(self, losses, pl_module, **kwargs):
        # make sure losses and task_weights are on the same device
        if self.task_weights.device != losses.device:
            self.task_weights = self.task_weights.to(losses.device)

        # compute tot_loss
        tot_loss = torch.sum(losses * self.task_weights)

        # backward
        pl_module.manual_backward(tot_loss)

        self.tot_loss = tot_loss


class AdaMt(WeightingMethod):
    '''Task weights are determined by the gradient magnitude of task-specific params. 

    Jha et al., (2020): AdaMT-Net: An Adaptive Weight Learning Based Multi-Task Learning Model For Scene Understanding
    '''

    def __init__(self, n_tasks, **kwargs):
        self.n_tasks = n_tasks
        self.task_weights = torch.ones(n_tasks)/n_tasks

    def backward(self, losses, pl_module, **kwargs):
        if self.task_weights.device != losses.device:
            self.task_weights = self.task_weights.to(losses.device)

        # update task_weights
        task_specific_params = pl_module.model.get_task_specific_params()
        task_specific_grads = [
            torch.autograd.grad(l, task_specific_params, retain_graph=True)
            for l in losses
        ]
        task_specific_norms = torch.stack([
            torch.linalg.norm(self._flattening(g))
            for g in task_specific_grads])  # shape (n_tasks,)

        assert len(
            task_specific_norms) == self.n_tasks, f'Require len(task_specific_norms)==n_tasks, but got {task_specific_norms}'

        self.task_weights = task_specific_norms/task_specific_norms.sum()

        # backward
        tot_loss = torch.sum(losses * self.task_weights)
        pl_module.manual_backward(tot_loss)

        self.tot_loss = tot_loss

    @staticmethod
    def _flattening(grad):
        return torch.cat(tuple(g.reshape(-1, ) for i, g in enumerate(grad)), axis=0)


class Dwa(WeightingMethod):
    '''DynamicWeightAverage
    Task losses (loss_i) are averaged across every EPOCH (not several batches)!
    '''

    def __init__(self, n_tasks, tempreture, max_epochs, **kwargs):

        self.tempreture = tempreture
        self.avg_losses = torch.zeros(max_epochs, n_tasks, dtype=torch.float)
        self.task_weights = torch.ones(n_tasks, dtype=torch.float)

    def backward(self, losses, pl_module, **kwargs):
        # send tensors to the devices they belong to
        if self.avg_losses.device != losses.device:
            self.avg_losses = self.avg_losses.to(pl_module.device)

        if self.task_weights.device != losses.device:
            self.task_weights = self.task_weights.to(pl_module.device)

        # get num of batches in every epoch
        epoch = pl_module.current_epoch
        n_train_batches = len(pl_module.trainer.train_dataloader)

        # update avg_losses
        costs = losses.detach()
        self.avg_losses[epoch, :] += costs / n_train_batches

        # update task_weights
        # - in the first epoch don't update task_weights
        if epoch >= 1:
            v = self.avg_losses[epoch-1, :] / self.avg_losses[epoch-2, :]
            self.task_weights = F.softmax(v/self.tempreture, dim=-1)

        # backward
        tot_loss = torch.sum(losses * self.task_weights)
        pl_module.manual_backward(tot_loss)

        self.tot_loss = tot_loss


class Uncert(WeightingMethod):
    def __init__(self, uncert_mode, **kwargs):
        self.uncert_mode = uncert_mode

    def backward(self, losses, pl_module, **kwargs):
        # calc tot_loss
        precision = torch.exp(-pl_module.logvars)

        if self.uncert_mode == 'no_plus1':
            tot_loss = torch.sum(losses * precision + pl_module.logvars)
        elif self.uncert_mode == 'plus1':
            tot_loss = torch.sum(losses * precision +
                                 torch.log(1+torch.exp(pl_module.logvars)))

        # backward
        pl_module.manual_backward(tot_loss)

        self.tot_loss = tot_loss

        # log precision and logvars
        for t, p, v in zip(pl_module.hparams.datamodule_cfg.tasks, precision, pl_module.logvars):
            pl_module.log(f'train/task_precision_{t}', p)
            pl_module.log(f'train/task_logvar_{t}', v)


class GradCos(WeightingMethod):
    def __init__(self, **kwargs):
        pass

    def backward(self, losses, pl_module, shared_params, **kwargs):
        # must come before manual_backward; otherwise will get
        # an error "Trying to backward through the graph a second time"
        shared_grads = self.get_shared_grads(
            losses,
            shared_params=shared_params
        )

        # calc grads for task-specific parameters
        tot_loss = torch.sum(losses)
        pl_module.manual_backward(tot_loss)

        self.tot_loss = tot_loss

        # update grads for shared parameters
        for p, g in zip(shared_params, shared_grads):
            p.grad = g

    def get_shared_grads(self, losses, shared_params):
        '''Calc gradidents for shared parameters
        '''

        assert len(
            losses) >= 2, f'Require number of losses is at least two when using GradCos! Got {losses}'

        pri_loss = losses[0]
        aux_losses = losses[1:]

        pri_grad = torch.autograd.grad(
            pri_loss, shared_params, retain_graph=True)

        # init shard_grads
        shared_grads = tuple(g.clone() for g in pri_grad)

        for loss in aux_losses:
            aux_grad = torch.autograd.grad(
                loss, shared_params, retain_graph=True)
            cosine = self.get_grad_cos_sim(pri_grad, aux_grad)

            if cosine > 0:
                shared_grads = tuple(
                    g + ga for g, ga in zip(shared_grads, aux_grad))

        return shared_grads

    @staticmethod
    def _flattening(grad):
        return torch.cat(tuple(g.reshape(-1, ) for i, g in enumerate(grad)), axis=0)

    def get_grad_cos_sim(self, grad1, grad2):
        '''Computes cosine simillarity of gradients after flattening of tensors.
        '''

        flat_grad1 = self._flattening(grad1)
        flat_grad2 = self._flattening(grad2)

        cosine = torch.nn.CosineSimilarity(dim=0)(flat_grad1, flat_grad2)

        return torch.clamp(cosine, -1, 1)


class GradNorm(WeightingMethod):
    '''Must be used with DDP, since DeepSpeed doesn't support
    "retain_graph"
    '''

    def __init__(self, n_tasks, alpha=1.5, **kwargs):
        self.n_tasks = n_tasks
        self.alpha = alpha
        self.init_losses = None
        self.task_weights = None  # it's just a copy of the true task_weights from the pl_module

    def backward(self, losses, pl_module, shared_params, **kwargs):
        # renormalize task_weights to sum up to 1
        pl_module.task_weights = torch.nn.Parameter(pl_module.task_weights.clamp(0, 1))
        task_weights_detached = pl_module.task_weights.detach()
        pl_module.task_weights = torch.nn.Parameter(task_weights_detached/task_weights_detached.sum())

        # the l_0 in the paper
        if self.init_losses is None:
            self.init_losses = losses.detach()

        # calc tot_loss
        tot_loss = torch.sum(pl_module.task_weights * losses)
        self.tot_loss = tot_loss

        # compute and retain gradients
        pl_module.manual_backward(tot_loss, retain_graph=True)

        # zero the w_i(t) gradients since we want to update the weights using gradnorm loss
        if pl_module.task_weights.grad is not None:
            pl_module.task_weights.grad = 0.0 * pl_module.task_weights.grad

        # compute grad norms
        norms = []
        for w_i, L_i in zip(pl_module.task_weights, losses):
            dlidW = torch.autograd.grad(
                L_i, shared_params, retain_graph=True)[0]
            norms.append(torch.norm(w_i * dlidW))

        norms = torch.stack(norms)

        # compute the constant term without accumulating gradients
        # as it should stay constant during back-propagation
        with torch.no_grad():
            # loss ratios
            loss_ratios = losses / self.init_losses
            # inverse training rate r(t)
            inverse_train_rates = loss_ratios / loss_ratios.mean()
            constant_term = norms.mean() * (inverse_train_rates ** self.alpha)

        grad_norm_loss = (norms - constant_term).abs().sum()
        pl_module.task_weights.grad = torch.autograd.grad(
            grad_norm_loss, pl_module.task_weights)[0]
        

        # just for logging purpose
        # make sure sum_i w_i = 1
        
        self.task_weights = task_weights_detached/(task_weights_detached.sum())


class OlAux(WeightingMethod):
    '''Paper Lin et al., (2019) `Adaptive Auxiliary Task Weighting for Reinforcement Learning`
    '''

    def __init__(self, n_tasks, N=5, w_lr=1, **kwargs):
        '''
        Args:
            N: the update batch size
            w_lr: learning rate for task weights
        '''
        self.N = N
        self.recent_pri_grads = []  # latest N batch primary grads
        self.recent_aux_grads = []  # latest N batch aux grads
        self.aux_task_weights = torch.ones(n_tasks-1)/(n_tasks-1)
        self.w_lr = w_lr

    def backward(self, losses, pl_module, shared_params, **kwargs):
        lr = pl_module.hparams.model_cfg.doc_encoder_lr

        if self.aux_task_weights.device != losses.device:
            self.aux_task_weights = self.aux_task_weights.to(losses.device)

        # -------------------
        # get losses
        # -------------------

        pri_loss = torch.log(losses[0])
        aux_losses = torch.log(losses[1:])

        # -------------------
        # update task_weights
        # -------------------

        # d(L_pri)/d(theta)
        pri_grads = torch.autograd.grad(
            pri_loss, shared_params, retain_graph=True,
            allow_unused=True)
        # Note:
        # we're not using pri_grad[0] because it's reserved for type token embeddings.
        # Since we're not using type token embeddings, its gradient will be None, which 
        # raises an error.
        pri_grads = self._flattening(tuple(g.clone() for g in pri_grads))

        self.recent_pri_grads.append(pri_grads)
        self.recent_pri_grads = self.recent_pri_grads[-self.N:]

        # d(L_aux)/d(theta)
        aux_grads = [
            torch.autograd.grad(
                aux_l, shared_params, retain_graph=True, allow_unused=True)
            for aux_l in aux_losses]
        aux_grads = [
            self._flattening(tuple(g.clone() for g in aux_g))
            for aux_g in aux_grads]
        aux_grads = torch.stack(aux_grads)  # shape (n_params, n_tasks)

        self.recent_aux_grads.append(aux_grads)
        self.recent_aux_grads = self.recent_aux_grads[-self.N:]

        # update weights every N batches
        if (pl_module.global_step >= self.N) and (pl_module.global_step % self.N == 0):
            task_weights_grads = [
                torch.matmul(aux_g, pri_g)
                for pri_g, aux_g in zip(self.recent_pri_grads, self.recent_aux_grads)]

            task_weights_grads = torch.stack(
                task_weights_grads).sum(dim=0).to(pl_module.device)  # (n_tasks-1,)

            # update aux task weights
            self.aux_task_weights += self.w_lr * lr * task_weights_grads

            # add primary weight
            self.task_weights = [1] + self.aux_task_weights.tolist()

        # -------------------
        # update parameters
        # -------------------

        tot_loss = torch.sum(
            pri_loss + torch.sum(aux_losses * self.aux_task_weights))

        # backward for parameters
        pl_module.manual_backward(tot_loss)

        self.tot_loss = tot_loss

    @staticmethod
    def _flattening(grad):
        return torch.cat(tuple(g.reshape(-1, ) for i, g in enumerate(grad)), axis=0)


class AuxiLearn(WeightingMethod):
    '''Navon et al., (2021)
    '''

    def __init__(self, **kwargs):
        pass

    def backward(self, losses, pl_module, **kwargs):
        pass


class GradPerp(WeightingMethod):
    '''Our proposal!

    Args:
        qr_mode: [diag/row/cos]. "diag" use the diagnal of the R matrix as
            task weights. "row" use rowsum(abs(R)) as task weights.
            "cos" use the cosine similarity between grad_pri
            and grad_aux as the task weights. "cos" is only used for
            illustration of the toy example
        M: multiplier of the primary task
        beta: smoothness of EMA
    '''

    def __init__(self, n_tasks, qr_mode, normalize_G, max_epochs, M, beta1=None, beta2=None, N=None, **kwargs):
        '''
        Args:
            K: primary task multiplier. A multiplier applied to 
               the pri_task_weight
            N: number of batches/epochs for averaging
            M: -1: learnable, >0: fixed
            qr_mode: "diag" or "row"
            beta1: 1st order grad smooth in EMA
            beta2: 2nd order grad smooth in EMA
        '''

        # set M if it's fixed
        if M == -1:
            self.M = None  # learnable
        elif M > 0:
            self.M = M  # fixed

        self.n_tasks = n_tasks
        self.normalize_G = normalize_G
        self.beta1 = beta1
        self.beta2 = beta2

        # internal, uncorrected G/task_weights
        self.avg_G = None
        self.avg_G_sq = None
        self.avg_task_weights = None

        self.qr_mode = qr_mode

        self.task_weights = torch.ones(n_tasks)/n_tasks
        # self.task_weights[0] *= M
        self.task_weights = self.task_weights/self.task_weights.sum()

        self.max_epochs = max_epochs
        self.last_epoch = None

    def backward(self, losses, pl_module, shared_params, **kwargs):
        if self.task_weights.device != losses.device:
            self.task_weights = self.task_weights.to(losses.device)

        # get global_step
        global_step = pl_module.global_step

        # wrap losses with log to scale
        losses = torch.log(losses)

        # times the primary loss by M
        if self.M == -1:
            losses[0] *= pl_module.model.M**2  # M is learnable
        else:
            losses[0] *= self.M  # M is fixed

        # ------
        # get G
        # ------
        G, G_sq = self.get_G(losses, shared_params=shared_params)

        # -------------------
        # smooth G (disabled)
        # -------------------
        # 1) updated (uncorrected, smoothed) self.G and self.G_avg
        # 2) return smoothed G for QR decomp
        # G = self._avg_G(
        #     G, G_sq,
        #     global_step=pl_module.global_step,
        #     epoch=pl_module.current_epoch,
        #     n_train_batches=len(pl_module.trainer.train_dataloader))

        '''
        # Debug: print cos similarity
        A = G[:,1:]
        t = G[:,0:1]
        y = A @ torch.linalg.lstsq(A, t).solution
        residual = y - t

        M = residual.norm()/y.norm()
        self.task_weights[0] = M

        # print(self.task_weights)

        pl_module.log('train/residual_over_t', M)
        '''

        # -------------------------------------
        # update task_weights (QR decomp)
        # -------------------------------------

        # get M (if M is learnable)
        # assert hasattr(pl_module.model, 'M'), 'pl_module.M is not found'

        # get task_weights

        # smooth task_weights
        # if global_step >= 100:
        if global_step >= 10:

            task_weights = self._qr(G, pl_module)

            self.task_weights = self._avg_task_weights(
                task_weights, global_step)

            # # sync (all_gather) task_weights
            # synced_task_weights = pl_module.all_gather(self.task_weights).detach()

            # print(f'{type(synced_task_weights)=}, {synced_task_weights=}')

            # self.task_weights = synced_task_weights.mean()

        # ---------------------------
        # backward for model params
        # ---------------------------
        self.task_weights /= self.task_weights.sum()

        tot_loss = torch.sum(losses * self.task_weights)
        self.tot_loss = tot_loss

        pl_module.manual_backward(tot_loss) 

    def get_G(self, losses, shared_params):
        '''For every loss_i, calc d(loss_i)/d(shred_params)
        '''

        assert len(losses) >= 2, \
            f'Require number of losses is at least two when using GradCos! Got {losses}'

        # prepare grads for QR decomposition
        #   - G: shape (n_params, n_tasks). The first col is always the pri
        G = [
            self._flattening(torch.autograd.grad(
                l, shared_params, retain_graph=True))
            for l in losses
        ]

        # normalize G so that each col is of unit length
        if self.normalize_G:
            G = [g / torch.linalg.norm(g) for g in G]

        G = torch.stack(G, dim=1).detach().float()
        G_sq = G**2

        return G, G_sq

    def _avg_G(self, G, G_sq, global_step, epoch, n_train_batches):
        '''
        1) updated (uncorrected, smoothed) self.G and self.G_avg
        2) return smoothed G for QR decomp
        '''
        # update 1st order G
        if self.avg_G is None:
            self.avg_G = G
        else:
            self.avg_G = (1-self.beta1) * G + \
                self.beta1 * self.avg_G

        avg_G_corrected = (1/(1-self.beta1**(global_step+1))) * self.avg_G

        # update 2nd order G (if needed)
        if self.beta2 is not None:
            if self.avg_G_sq is None:
                self.avg_G_sq = G_sq
            else:
                self.avg_G_sq = (1-self.beta2) * G_sq + \
                    self.beta2 * self.avg_G_sq

            avg_G_sq_corrected = (
                1/(1-self.beta2**(global_step+1))) * self.avg_G_sq

        # return final G for QR decomp
        if self.beta2 is not None:
            # if (global_step+1) % 10 == 0:
            #     torch.save(
            #         avg_G_corrected, f'/home/yu/OneDrive/CC/local-dev/data/gradients/{global_step}.pt')

            return avg_G_corrected/(torch.sqrt(avg_G_sq_corrected)+1e-8)
        else:
            return avg_G_corrected

    def _avg_task_weights(self, task_weights, global_step):
        if self.avg_task_weights is None:
            self.avg_task_weights = task_weights
        else:
            self.avg_task_weights = (1-self.beta1) * task_weights \
                + self.beta1 * self.avg_task_weights

        return (1/(1-self.beta1**(global_step+1))) * self.avg_task_weights

    def _avg_losses(self, losses, global_step):

        if self.avg_losses is None:
            self.avg_losses = losses
        else:
            self.avg_losses = (1-self.beta1) * losses \
                + self.beta1 * self.avg_losses

        return (1/(1-self.beta1**(global_step+1))) * self.avg_losses

    def _qr(self, G, pl_module):
        '''QR decomposition of the shared_grads

        Task weights are determined by the diagonal of the "r" matrix
        '''
        if self.qr_mode in ['diag', 'offdiag', 'relative-diag', 'relative-offdiag', 'row']:
            _, r = torch.linalg.qr(G, mode='r')

            # diag: abs value of the diagonal
            # offdiag: colsum of abs value of the off-diag
            # relative-offdiag: relative colsum of abs value of the off-diag
            # relative-diag: diag/sum(diag)
            # row: abs value of the rowsum
            if self.qr_mode == 'diag':
                task_weights = torch.abs(torch.diag(r))
            if self.qr_mode == 'offdiag':
                abs_r = torch.abs(r)
                colsum = abs_r.sum(0)
                task_weights = colsum - torch.diag(abs_r)
                task_weights[0] = colsum[0]
            elif self.qr_mode == 'row':
                task_weights = torch.abs(r).sum(dim=1)
            elif self.qr_mode == 'relative-diag':
                abs_r = torch.abs(r)
                task_weights = torch.diag(abs_r)/abs_r.sum(0)
            elif self.qr_mode == 'relative-offdiag':
                abs_r = torch.abs(r)
                colsum = abs_r.sum(0)
                off_diagsum = colsum - torch.diag(abs_r)
                task_weights = off_diagsum/colsum
                task_weights[0] = 1

            assert (task_weights.dim() == 1) and (
                task_weights.shape[0] == self.n_tasks), f'Require task_weights to be a vector, got {task_weights=}'

        # only for the toy example!
        elif self.qr_mode == 'cos':
            task_weights = []

            for col_idx in range(G.shape[1]):
                cos_score = self._get_grad_cos_sim(
                    G[:, 0], G[:, col_idx])

                # cos_score = 0 if cos_score < 0 else cos_score
                cos_score = torch.abs(cos_score)

                task_weights.append(cos_score)

            task_weights = torch.tensor(task_weights, device=G.device)

        else:
            raise Exception(f'Unrecognized qr_mode: {self.qr_mode}')

        # apply primary task multiplier
        # task_weights[0] *= self.M**2
        # task_weights[0] *= pl_module.model.M**2
        # task_weights[0] *= self.M

        # return task_weights/task_weights.sum()
        return task_weights

    @staticmethod
    def _flattening(grad):
        return torch.cat(tuple(g.reshape(-1, ) for i, g in enumerate(grad)), axis=0)

    def _get_grad_cos_sim(self, grad1, grad2):
        '''Computes cosine simillarity of gradients
        '''

        cosine = torch.nn.CosineSimilarity(dim=0)(grad1, grad2)

        return torch.clamp(cosine, -1, 1)
