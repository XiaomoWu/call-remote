import torch
import torch.nn.functional as F
import torchmetrics
from omegaconf import ListConfig


class Fixed(torchmetrics.Metric):
    def __init__(self, n_tasks, init_task_weights,
                 compute_on_step=True,
                 dist_sync_on_step=False,
                 squared=True,
                 process_group=None,
                 dist_sync_fn=None):
        '''
        Given (possibly multiple) t and y, compute weighted rmse
        '''

        super().__init__(compute_on_step=compute_on_step,
                         dist_sync_on_step=dist_sync_on_step, process_group=process_group,
                         dist_sync_fn=dist_sync_fn)

        # Save weights
        if isinstance(init_task_weights, ListConfig):
            init_task_weights = torch.tensor(init_task_weights)

        self.register_buffer('task_weights', init_task_weights)

        # Add states
        self.add_state('sum_squared_error', default=torch.zeros(
            n_tasks), dist_reduce_fx='sum')
        self.add_state("total", default=torch.zeros(
            n_tasks), dist_reduce_fx='sum')

        self.squared = squared

    def _mean_squared_error_update(self, y, t):
        assert y.shape == t.shape
        assert y.dim() == 2

        diff = y - t
        sum_squared_error = torch.sum(diff * diff, dim=0)
        n_obs = y.size()[0]  # 0-dim

        return sum_squared_error, n_obs

    def _mean_squared_error_compute(self, sum_squared_error, n_obs, task_weights, squared=True):
        mse = (sum_squared_error / n_obs) * task_weights
        mse = torch.sum(mse)

        return mse if squared else torch.sqrt(mse)

    def update(self, y, t, **kwargs):
        # update for MSE
        sum_squared_error, n_obs = self._mean_squared_error_update(y, t)
        self.sum_squared_error += sum_squared_error
        self.total += n_obs

    def compute(self):
        tot_mse = self._mean_squared_error_compute(
            self.sum_squared_error, self.total, self.task_weights, squared=self.squared)

        return tot_mse


class Dwa(torchmetrics.Metric):
    def __init__(self, n_tasks,
                 dwa_mode,
                 beta,
                 tempreture=None,
                 compute_on_step=True,
                 dist_sync_on_step=False,
                 squared=True,
                 process_group=None,
                 dist_sync_fn=None,
                 ):
        '''
        Output:
          - total loss
          - a vector of task loss
        '''

        super().__init__(compute_on_step=compute_on_step,
                         dist_sync_on_step=dist_sync_on_step, process_group=process_group,
                         dist_sync_fn=dist_sync_fn)

        # hparam sanity check
        if dwa_mode == 'softmax':
            assert tempreture is not None, f'Require tempreture to be NOT None when dwa_mode=softmax!'
        if dwa_mode == 'frac':
            assert tempreture is None, f'Require tempreture to be None when dwa_mode=frac!'

        # init task weights
        self.register_buffer('task_weights', torch.ones(n_tasks)/n_tasks)

        # init task losses and loss decreasing speed
        # - v: the vector of loss decreasing speed (r=L(t)/L(t-1))
        # - we must store two versions of v for bias correction
        self.register_buffer('last_task_losses',
                             torch.tensor([float('inf')]*n_tasks))
        self.register_buffer('uncorrected_v', torch.zeros(n_tasks))
        self.register_buffer('v', torch.zeros(n_tasks))

        # add states
        self.add_state('sum_squared_error', default=torch.zeros(
            n_tasks), dist_reduce_fx='sum', persistent=True)
        self.add_state("total", default=torch.zeros(
            n_tasks), dist_reduce_fx='sum', persistent=True)

        # add other states
        self.squared = squared
        self.beta = beta
        self.tempreture = tempreture
        self.n_tasks = n_tasks
        self.dwa_mode = dwa_mode

    def _mean_squared_error_update(self, y, t):
        assert y.shape == t.shape
        assert y.dim() == 2

        diff = y - t
        sum_squared_error = torch.sum(diff * diff, dim=0)
        n_obs = y.size()[0]

        return sum_squared_error, n_obs

    def _mean_squared_error_compute(self, sum_squared_error, n_obs, task_weights, squared=True):
        task_losses = sum_squared_error / n_obs
        total_loss = torch.sum(task_losses * task_weights)

        return (total_loss if squared else torch.sqrt(total_loss),
                task_losses.detach_())

    def update(self, y, t, global_step, **kwargs):
        # update for MSE
        sum_squared_error, n_obs = self._mean_squared_error_update(y, t)
        self.sum_squared_error += sum_squared_error
        self.total += n_obs
        self.global_step = global_step

    def compute(self):
        '''
        Note:
            Every process will has its own task_weights and they won't be
            synced!
        '''
        # compute total mse
        total_loss, task_losses = self._mean_squared_error_compute(
            self.sum_squared_error, self.total, self.task_weights, squared=self.squared)

        with torch.no_grad():
            # update v (training speed)
            # - only update when global_step >= 1 (the first step is 0)
            current_v = task_losses / self.last_task_losses
            if self.global_step >= 1:
                self.uncorrected_v = (
                    self.beta*self.uncorrected_v + (1-self.beta)*current_v)
                self.v = self.uncorrected_v / (1-self.beta**self.global_step)

            # update task weights
            if self.dwa_mode == 'softmax':
                self.task_weights = F.softmax(self.v/self.tempreture, dim=-1)
            elif self.dwa_mode == 'frac':
                # don't update in first step (step=0)
                if self.global_step >= 1:
                    self.task_weights = self.v / self.v.sum()

            # update last_task_losses
            self.last_task_losses = task_losses

        # compute r
        return total_loss


class AdaMt(torchmetrics.Metric):
    '''task_weights are determined by the magnitude of grad
    '''

    def __init__(self, n_tasks,
                 beta,
                 tempreture,
                 adamt_mode,
                 compute_on_step=True,
                 dist_sync_on_step=False,
                 squared=True,
                 process_group=None,
                 dist_sync_fn=None,
                 ):
        '''
        AdaMT-Net (Jha et al., 2021)
        '''

        super().__init__(compute_on_step=compute_on_step,
                         dist_sync_on_step=dist_sync_on_step, process_group=process_group,
                         dist_sync_fn=dist_sync_fn)
        # init task weights
        self.register_buffer('task_weights', torch.ones(n_tasks)/n_tasks)

        self.register_buffer("uncorrected_phi_grads",
                             torch.zeros(n_tasks, 1))
        self.register_buffer("phi_grads", torch.zeros(n_tasks, 1))

        # Add states
        self.add_state('sum_squared_error', default=torch.zeros(
            n_tasks), dist_reduce_fx='sum')
        self.add_state("total", default=torch.zeros(
            n_tasks), dist_reduce_fx='sum')

        self.squared = squared
        self.beta = beta
        self.n_tasks = n_tasks
        self.tempreture = tempreture
        self.adamt_mode = adamt_mode

    def _mean_squared_error_update(self, y, t):
        assert y.shape == t.shape
        assert y.dim() == 2

        diff = y - t
        sum_squared_error = torch.sum(diff * diff, dim=0)
        n_obs = y.size()[0]  # 0-dim

        return sum_squared_error, n_obs

    def _mean_squared_error_compute(self, sum_squared_error, n_obs, task_weights, squared=True):
        mse = (sum_squared_error / n_obs) * task_weights
        mse = torch.sum(mse)

        return mse if squared else torch.sqrt(mse)

    def update(self, y, t, phi_grads, global_step, **kwargs):
        # update for MSE
        sum_squared_error, n_obs = self._mean_squared_error_update(y, t)
        self.sum_squared_error += sum_squared_error
        self.total += n_obs
        self.global_step = global_step

        # update grads
        # Attention 1
        #   Unlike DWA where task_losses are different across GPUs,
        #   the task_grads in AdaMT are the SAME across GPUs. It's
        #   because grads are synced during backprop.
        # Attention 2
        #   Torchmetrics will call "update" twiece in a single forward
        #   step. To avoid the task_grads being averaged twice, I only
        #   update it when self._to_sync is True.
        if (phi_grads is not None) and (global_step >= 1) and self._to_sync:
            # mask out all nan or inf in task_grads
            current_phi_grads = torch.nan_to_num(phi_grads)
            self.uncorrected_phi_grads = self.beta * \
                self.uncorrected_phi_grads + (1-self.beta)*current_phi_grads
            self.phi_grads = self.uncorrected_phi_grads / \
                (1-self.beta**global_step)

    def compute(self):
        # compute grad_norm and update task_weights
        # Note 1:
        #   I'm not using softmax to get the task_weights because
        #   the maganitude of task_grad_norms is so small that
        #   and the results of softmaxing is always [1, 1]
        # Note 2:
        #   Must enable FP32 because FP16 is numerically instable!
        phi_grad_norms = torch.linalg.norm(self.phi_grads, dim=-1)

        if self.adamt_mode == 'frac':
            if self.global_step >= 1:
                self.task_weights = phi_grad_norms/phi_grad_norms.sum()
        elif self.adamt_mode == 'softmax':
            self.task_weights = F.softmax(
                phi_grad_norms/self.tempreture, dim=-1)
        else:
            raise Exception('Unkown adamt_mode: {self.adamt_mode}')

        # update total_loss
        tot_mse = self._mean_squared_error_compute(
            self.sum_squared_error, self.total, self.task_weights, squared=self.squared)

        return tot_mse


class Sigma(torchmetrics.Metric):
    '''task_weights are determined by task uncertainty
    '''

    def __init__(self, n_tasks,
                 sigma_mode,
                 compute_on_step=True,
                 dist_sync_on_step=False,
                 squared=True,
                 process_group=None,
                 dist_sync_fn=None):

        super().__init__(compute_on_step=compute_on_step,
                         dist_sync_on_step=dist_sync_on_step, process_group=process_group,
                         dist_sync_fn=dist_sync_fn)

        # Add states
        self.add_state('sum_squared_error', default=torch.zeros(
            n_tasks), dist_reduce_fx='sum')
        self.add_state("total", default=torch.zeros(
            n_tasks), dist_reduce_fx='sum')

        self.squared = squared
        self.sigma_mode = sigma_mode

    def _mean_squared_error_update(self, y, t):
        assert y.shape == t.shape
        assert y.dim() == 2

        diff = y - t
        sum_squared_error = torch.sum(diff * diff, dim=0)
        n_obs = y.size()[0]  # 0-dim

        return sum_squared_error, n_obs

    def _mean_squared_error_compute(self, sum_squared_error, n_obs, sigmas, squared=True):
        # Atten!
        #   I'm not using task_weights, instead, the total MSE is
        #   computed as:
        #      \sum_i {exp(-sigma_i) * loss_i + sigma_i}
        precision = torch.exp(-sigmas)

        if self.sigma_mode == 'no_plus1':
            mse = (sum_squared_error / n_obs) * precision + sigmas
        elif self.sigma_mode == 'plus1':
            mse = (sum_squared_error / n_obs) * \
                precision + torch.log(1+torch.exp(sigmas))
        else:
            raise Exception(f'Unknown sigma_mode: {self.sigma_mode}')

        mse = torch.sum(mse)

        # store precision for logging
        self.precision = precision

        return mse if squared else torch.sqrt(mse)

    def update(self, y, t, sigmas, **kwargs):
        # update for MSE
        sum_squared_error, n_obs = self._mean_squared_error_update(y, t)
        self.sum_squared_error += sum_squared_error
        self.total += n_obs
        self.sigmas = sigmas

    def compute(self):
        tot_mse = self._mean_squared_error_compute(
            self.sum_squared_error, self.total, self.sigmas, squared=self.squared)

        return tot_mse


class Cos(torchmetrics.Metric):
    '''Task_weights are determined by cosine similarity between grads
    (Du et al., 2018) Adapting Auxiliary losses Using Gradient Similarity
    '''

    def __init__(self, n_tasks,
                 cos_mode,
                 compute_on_step=True,
                 dist_sync_on_step=False,
                 squared=True,
                 process_group=None,
                 dist_sync_fn=None,
                 ):
        '''
        Cosine similarity (Du et al., 2018)
        '''

        super().__init__(compute_on_step=compute_on_step,
                         dist_sync_on_step=dist_sync_on_step, process_group=process_group,
                         dist_sync_fn=dist_sync_fn)
        # init task weights
        self.register_buffer('task_weights', torch.ones(n_tasks)/n_tasks)

        self.register_buffer("uncorrected_phi_grads",
                             torch.zeros(n_tasks, 1))
        self.register_buffer("phi_grads", torch.zeros(n_tasks, 1))

        # Add states
        self.add_state('sum_squared_error', default=torch.zeros(
            n_tasks), dist_reduce_fx='sum')
        self.add_state("total", default=torch.zeros(
            n_tasks), dist_reduce_fx='sum')

        self.squared = squared
        self.n_tasks = n_tasks
        self.cos_mode = cos_mode
        self.beta = 0.9

    def _mean_squared_error_update(self, y, t):
        assert y.shape == t.shape
        assert y.dim() == 2

        diff = y - t
        sum_squared_error = torch.sum(diff * diff, dim=0)
        n_obs = y.size()[0]  # 0-dim

        return sum_squared_error, n_obs

    def _mean_squared_error_compute(self, sum_squared_error, n_obs, task_weights, squared=True):
        mse = (sum_squared_error / n_obs) * task_weights
        mse = torch.sum(mse)

        return mse if squared else torch.sqrt(mse)

    def update(self, y, t, phi_grads, global_step, **kwargs):
        # update for MSE
        sum_squared_error, n_obs = self._mean_squared_error_update(y, t)
        self.sum_squared_error += sum_squared_error
        self.total += n_obs
        self.global_step = global_step

        # update grads
        # Attention 1
        #   Unlike DWA where task_losses are different across GPUs,
        #   the task_grads in AdaMT are the SAME across GPUs. It's
        #   because grads are synced during backprop.
        # Attention 2
        #   Torchmetrics will call "update" twiece in a single forward
        #   step. To avoid the task_grads being averaged twice, I only
        #   update it when self._to_sync is True.
        if (phi_grads is not None) and (global_step >= 1) and self._to_sync:
            # mask out all nan or inf in task_grads
            current_phi_grads = torch.nan_to_num(phi_grads)
            self.uncorrected_phi_grads = self.beta * \
                self.uncorrected_phi_grads + (1-self.beta)*current_phi_grads
            self.phi_grads = self.uncorrected_phi_grads / \
                (1-self.beta**global_step)

    def compute(self):
        # compute grad_norm and update task_weights
        # Note 1:
        #   I'm not using softmax to get the task_weights because
        #   the maganitude of task_grad_norms is so small that
        #   and the results of softmaxing is always [1, 1]
        # Note 2:
        #   Must enable FP32 because FP16 is numerically instable!
        phi_grad_norms = torch.linalg.norm(self.phi_grads, dim=-1)

        # use frac to compute task_weights
        if self.global_step >= 1:
            self.task_weights = phi_grad_norms/phi_grad_norms.sum()

        # update total_loss
        tot_mse = self._mean_squared_error_compute(
            self.sum_squared_error, self.total, self.task_weights, squared=self.squared)

        return tot_mse
