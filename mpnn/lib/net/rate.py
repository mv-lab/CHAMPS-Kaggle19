# learning rate schduler
from lib.include import *

# http://elgoacademy.org/anatomy-matplotlib-part-1/
def plot_rates(fig, lrs, title=''):

    N = len(lrs)
    epoches = np.arange(0,N)


    #get limits
    max_lr  = np.max(lrs)
    xmin=0
    xmax=N
    dx=2

    ymin=0
    ymax=max_lr*1.2
    dy=(ymax-ymin)/10
    dy=10**math.ceil(math.log10(dy))

    ax = fig.add_subplot(111)
    #ax = fig.gca()
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.set_xticks(np.arange(xmin,xmax+0.0001, dx))
    ax.set_yticks(np.arange(ymin,ymax+0.0001, dy))
    ax.set_xlim(xmin,xmax+0.0001)
    ax.set_ylim(ymin,ymax+0.0001)
    ax.grid(b=True, which='minor', color='black', alpha=0.1, linestyle='dashed')
    ax.grid(b=True, which='major', color='black', alpha=0.4, linestyle='dashed')

    ax.set_xlabel('iter')
    ax.set_ylabel('learning rate')
    ax.set_title(title)
    ax.plot(epoches, lrs)



## simple stepping rates
class StepScheduler():
    def __init__(self, pairs):
        super(StepScheduler, self).__init__()

        N=len(pairs)
        rates=[]
        steps=[]
        for n in range(N):
            steps.append(pairs[n][0])
            rates.append(pairs[n][1])

        self.rates = rates
        self.steps = steps

    def __call__(self, epoch):
        N = len(self.steps)
        lr = -1
        for n in range(N):
            if epoch >= self.steps[n]:
                lr = self.rates[n]
        return lr

    def __str__(self):
        string = 'Step Learning Rates\n' \
                + 'rates=' + str(['%7.5f' % i for i in self.rates]) + '\n' \
                + 'steps=' + str(['%7.0f' % i for i in self.steps]) + ''
        return string

    def step(self, epoch=None):
        pass


## https://github.com/pytorch/tutorials/blob/master/beginner_source/transfer_learning_tutorial.py
class DecayScheduler():
    def __init__(self, base_lr, decay, step):
        super(DecayScheduler, self).__init__()
        self.step  = step
        self.decay = decay
        self.base_lr = base_lr

    def get_rate(self, epoch):
        lr = self.base_lr * (self.decay**(epoch // self.step))
        return lr



    def __str__(self):
        string = '(Exp) Decay Learning Rates\n' \
                + 'base_lr=%0.3f, decay=%0.3f, step=%0.3f'%(self.base_lr, self.decay, self.step)
        return string




# 'Cyclical Learning Rates for Training Neural Networks'- Leslie N. Smith, arxiv 2017
#       https://arxiv.org/abs/1506.01186
#       https://github.com/bckenstler/CLR

class CyclicScheduler1():

    def __init__(self, min_lr=0.001, max_lr=0.01, period=10 ):
        super(CyclicScheduler, self).__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.period = period

    def __call__(self, time):

        #sawtooth
        #r = (1-(time%self.period)/self.period)

        #cosine
        time= time%self.period
        r = (np.cos(time/self.period *PI)+1)/2

        lr = self.min_lr + r*(self.max_lr-self.min_lr)
        return lr

    def __str__(self):
        string = 'CyclicScheduler\n' \
                + 'min_lr=%0.3f, max_lr=%0.3f, period=%8.1f'%(self.min_lr, self.max_lr, self.period)
        return string


class CyclicScheduler2():

    def __init__(self, min_lr=0.001, max_lr=0.01, period=10, max_decay=0.99, warm_start=0 ):
        super(CyclicScheduler, self).__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.period = period
        self.max_decay = max_decay
        self.warm_start = warm_start
        self.cycle = -1

    def __call__(self, time):
        if time<self.warm_start: return self.max_lr

        #cosine
        self.cycle = (time-self.warm_start)//self.period
        time = (time-self.warm_start)%self.period

        period = self.period
        min_lr = self.min_lr
        max_lr = self.max_lr *(self.max_decay**self.cycle)


        r   = (np.cos(time/period *PI)+1)/2
        lr = min_lr + r*(max_lr-min_lr)

        return lr



    def __str__(self):
        string = 'CyclicScheduler\n' \
                + 'min_lr=%0.4f, max_lr=%0.4f, period=%8.1f'%(self.min_lr, self.max_lr, self.period)
        return string


#tanh curve
class CyclicScheduler3():

    def __init__(self, min_lr=0.001, max_lr=0.01, period=10, max_decay=0.99, warm_start=0 ):
        super(CyclicScheduler, self).__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.period = period
        self.max_decay = max_decay
        self.warm_start = warm_start
        self.cycle = -1

    def __call__(self, time):
        if time<self.warm_start: return self.max_lr

        #cosine
        self.cycle = (time-self.warm_start)//self.period
        time = (time-self.warm_start)%self.period

        period = self.period
        min_lr = self.min_lr
        max_lr = self.max_lr *(self.max_decay**self.cycle)


        r   = (np.tanh(-time/period *16 +8)+1)*0.5
        lr = min_lr + r*(max_lr-min_lr)

        return lr



    def __str__(self):
        string = 'CyclicScheduler\n' \
                + 'min_lr=%0.3f, max_lr=%0.3f, period=%8.1f'%(self.min_lr, self.max_lr, self.period)
        return string

#
# class CyclicScheduler():
#
#     def __init__(self, pairs, period=10, max_decay=1, warm_start=0 ):
#         super(CyclicScheduler, self).__init__()
#
#         self.lrs=[]
#         self.steps=[]
#         for p in pairs:
#             self.steps.append(p[0])
#             self.lrs.append(p[1])
#
#
#         self.period = period
#         self.warm_start = warm_start
#         self.max_decay = max_decay
#         self.cycle = -1
#
#     def __call__(self, time):
#         if time<self.warm_start: return self.lrs[0]
#
#         self.cycle = (time-self.warm_start)//self.period
#         time = (time-self.warm_start)%self.period
#
#         rates = self.lrs.copy()
#         steps = self.steps
#         rates[0] = rates[0] *(self.max_decay**self.cycle)
#         lr = -1
#         for rate,step in zip(rates,steps):
#             if time >= step:
#                lr = rate
#
#         return lr
#
#
#
#     def __str__(self):
#         string = 'CyclicScheduler\n' \
#                 + 'lrs  =' + str(['%7.4f' % i for i in self.lrs]) + '\n' \
#                 + 'steps=' + str(['%7.0f' % i for i in self.steps]) + '\n' \
#                 + 'period=%8.1f'%(self.period)
#         return string


class NullScheduler():
    def __init__(self, lr=0.01 ):
        super(NullScheduler, self).__init__()
        self.lr    = lr
        self.cycle = 0

    def __call__(self, time):
        return self.lr

    def __str__(self):
        string = 'NullScheduler\n' \
                + 'lr=%0.5f '%(self.lr)
        return string

# net ------------------------------------
# https://github.com/pytorch/examples/blob/master/imagenet/main.py ###############
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_betas(optimizer, beta1, beta2):
    for param_group in optimizer.param_groups:
        print(param_group['betas'])
        param_group['betas'] = (beta1, beta2)

def adjust_max_learning_rate(optimizer, max_lr):
    for param_group in optimizer.param_groups:
        param_group['max_lr'] = max_lr

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]

    assert(len(lr)==1) #we support only one param_group
    lr = lr[0]

    return lr

class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
            last_epoch = 0
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_epoch = last_epoch

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `lr_scheduler.step()` is called after
        # `optimizer.step()`
        def with_counter(func, opt):
            def wrapper(*args, **kwargs):
                opt._step_count += 1
                return func(*args, **kwargs)
            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step, self.optimizer)
        self.optimizer._step_count = 0
        self._step_count = 0
        self.step(last_epoch)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn("Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                              "initialization. Please, make sure to call `optimizer.step()` before "
                              "`lr_scheduler.step()`. See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                              "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                              "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                              "will result in PyTorch skipping the first value of the learning rate schedule."
                              "See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
        self._step_count += 1

        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class OneCycleLR(_LRScheduler):
    """Sets the learning rate of each parameter group according to the
    1cycle learning rate policy. The 1cycle policy anneals the learning
    rate from an initial learning rate to some maximum learning rate and then
    from that maximum learning rate to some minimum learning rate much lower
    than the initial learning rate.
    This policy was initially described in the paper `Super-Convergence:
    Very Fast Training of Neural Networks Using Large Learning Rates`_.
    The 1cycle learning rate policy changes the learning rate after every batch.
    `step` should be called after a batch has been used for training.
    This class has two built-in annealing strategies:
    "cos":
        Cosine annealing
    "linear":
        Linear annealing
    Note also that the total number of steps in the cycle can be determined in one
    of two ways (listed in order of precedence):
    1) A value for total_steps is explicitly provided.
    2) A number of epochs (num_epochs) and a DataLoader object (train_dl) are provided.
       In this case, the number of total steps is inferred by
       total_steps = num_epochs * len(train_dl)
    You must either provide a value for total_steps or provide a value for both
    num_epochs and train_dl.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_lr (float or list): Upper learning rate boundaries in the cycle
            for each parameter group.
        total_steps (int): The total number of steps in the cycle. Note that
            if a value is provided here, then it must be inferred by providing
            a value for num_epochs and train_dl.
            Default: None
        num_epochs (int): The number of epochs to train for. This is used along
            with train_dl in order to infer the total number of steps in the cycle
            if a value for total_steps is not provided.
            Default: None
        train_dl (DataLoader): The DataLoader object to train over. This is used
            along with num_epochs in order to infer the total number of steps in
            the cycle if a value for total_steps is not provided.
            Default: None
        pct_start (float): The percentage of the cycle (in number of steps) spent
            increasing the learning rate.
            Default: 0.3
        anneal_strategy (str): {'cos', 'linear'}
            Specifies the annealing strategy.
            Default: 'cos'
        cycle_momentum (bool): If ``True``, momentum is cycled inversely
            to learning rate between 'base_momentum' and 'max_momentum'.
            Default: True
        base_momentum (float or list): Lower momentum boundaries in the cycle
            for each parameter group. Note that momentum is cycled inversely
            to learning rate; at the peak of a cycle, momentum is
            'base_momentum' and learning rate is 'max_lr'.
            Default: 0.85
        max_momentum (float or list): Upper momentum boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_momentum - base_momentum).
            Note that momentum is cycled inversely
            to learning rate; at the start of a cycle, momentum is 'max_momentum'
            and learning rate is 'base_lr'
            Default: 0.95
        div_factor (float): Determines the initial learning rate via
            initial_lr = max_lr/div_factor
            Default: 25
        final_div_factor (float): Determines the minimum learning rate via
            min_lr = initial_lr/final_div_factor
            Default: 1e4
        last_epoch (int): The index of the last batch. This parameter is used when
            resuming a training job. Since `step()` should be invoked after each
            batch instead of after each epoch, this number represents the total
            number of *batches* computed, not the total number of epochs computed.
            When last_epoch=-1, the schedule is started from the beginning.
            Default: -1
    Example:
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, train_dl=data_loader, num_epochs=10)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         train_batch(...)
        >>>         scheduler.step()
    .. _Super-Convergence\: Very Fast Training of Neural Networks Using Large Learning Rates:
        https://arxiv.org/abs/1708.07120
    """
    def __init__(self,
                 optimizer,
                 max_lr,
                 total_steps=None,
                 num_epochs=None,
                 train_dl=None,
                 pct_start=0.3,
                 anneal_strategy='cos',
                 cycle_momentum=True,
                 base_momentum=0.85,
                 max_momentum=0.95,
                 div_factor=25.,
                 final_div_factor=1e4,
                 last_epoch=-1):

        self.optimizer = optimizer

        # Validate total_steps
        if total_steps is None and num_epochs is None and train_dl is None:
            raise ValueError("You must define either total_steps OR (num_epochs AND train_dl)")
        elif total_steps is not None:
            if total_steps <= 0 or not isinstance(total_steps, int):
                raise ValueError("Expected non-negative integer total_steps, but got {}".format(total_steps))
            self.total_steps = total_steps
        else:
            if num_epochs <= 0 or not isinstance(num_epochs, int):
                raise ValueError("Expected non-negative integer num_epochs, but got {}".format(num_epochs))
            if not hasattr(train_dl, '__len__'):
                raise ValueError("len is not defined over given train_dl")
            self.total_steps = num_epochs * len(train_dl)
        self.step_size_up = float(pct_start * self.total_steps) - 1
        self.step_size_down = float(self.total_steps - self.step_size_up) - 1

        # Validate pct_start
        if pct_start < 0 or pct_start > 1 or not isinstance(pct_start, float):
            raise ValueError("Expected float between 0 and 1 pct_start, but got {}".format(pct_start))

        # Validate anneal_strategy
        if anneal_strategy not in ['cos', 'linear']:
            raise ValueError("anneal_strategy must by one of 'cos' or 'linear', instead got {}".format(anneal_strategy))
        elif anneal_strategy == 'cos':
            self.anneal_func = self._annealing_cos
        elif anneal_strategy == 'linear':
            self.anneal_func = self._annealing_linear

        # Initialize learning rate variables
        self.max_lr = max_lr
        max_lrs = self._format_param('max_lr', self.optimizer, max_lr)
        if last_epoch == -1:
            for idx, group in enumerate(self.optimizer.param_groups):
                group['lr'] = max_lrs[idx] / div_factor
                group['max_lr'] = max_lrs[idx]
                group['min_lr'] = group['lr'] / final_div_factor

        # Initialize momentum variables
        self.cycle_momentum = cycle_momentum
        if self.cycle_momentum:
            if 'momentum' not in self.optimizer.defaults and 'betas' not in self.optimizer.defaults:
                raise ValueError('optimizer must support momentum with `cycle_momentum` option enabled')
            self.use_beta1 = 'betas' in self.optimizer.defaults
            max_momentums = self._format_param('max_momentum', optimizer, max_momentum)
            base_momentums = self._format_param('base_momentum', optimizer, base_momentum)
            if last_epoch == -1:
                for m_momentum, b_momentum, group in zip(max_momentums, base_momentums, optimizer.param_groups):
                    if self.use_beta1:
                        _, beta2 = group['betas']
                        group['betas'] = (m_momentum, beta2)
                    else:
                        group['momentum'] = m_momentum
                    group['max_momentum'] = m_momentum
                    group['base_momentum'] = b_momentum

        super(OneCycleLR, self).__init__(optimizer, last_epoch)

    def _format_param(self, name, optimizer, param):
        """Return correctly formatted lr/momentum for each param group."""
        if isinstance(param, (list, tuple)):
            if len(param) != len(optimizer.param_groups):
                raise ValueError("expected {} values for {}, got {}".format(
                    len(optimizer.param_groups), name, len(param)))
            return param
        else:
            return [param] * len(optimizer.param_groups)

    def _annealing_cos(self, start, end, pct):
        "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end) / 2.0 * cos_out

    def _annealing_linear(self, start, end, pct):
        "Linearly anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        return (end - start) * pct + start

    def get_lr(self):
        lrs = []
        step_num = self.last_epoch

        if step_num > self.total_steps:
            raise ValueError("Tried to step {} times. The specified number of total steps is {}"
                             .format(step_num + 1, self.total_steps))

        for group in self.optimizer.param_groups:
            if step_num <= self.step_size_up:
                computed_lr = self.anneal_func(group['initial_lr'], group['max_lr'], step_num / self.step_size_up)
                if self.cycle_momentum:
                    computed_momentum = self.anneal_func(group['max_momentum'], group['base_momentum'],
                                                         step_num / self.step_size_up)
            else:
                down_step_num = step_num - self.step_size_up
                computed_lr = self.anneal_func(group['max_lr'], group['min_lr'], down_step_num / self.step_size_down)
                if self.cycle_momentum:
                    computed_momentum = self.anneal_func(group['base_momentum'], group['max_momentum'],
                                                         down_step_num / self.step_size_down)

            lrs.append(computed_lr)
            if self.cycle_momentum:
                if self.use_beta1:
                    _, beta2 = group['betas']
                    group['betas'] = (computed_momentum, beta2)
                else:
                    group['momentum'] = computed_momentum

        return lrs

    def __str__(self):
        string = 'OneCycleLR Scheduler\n' + '  start_lr=%0.5f \n  max_lr=%0.5f'%(self.get_lr()[0], self.max_lr)
        return string


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    num_iters=125

    scheduler = StepScheduler([ (0,0.1),  (10,0.01),  (25,0.005),  (35,0.001), (40,0.0001), (43,-1)])
    #scheduler = DecayScheduler(base_lr=0.1, decay=0.32, step=10)
    #scheduler = CyclicScheduler(min_lr=0.0001, max_lr=0.01, period=30., warm_start=5) ##exp_range ##triangular2
    #scheduler = CyclicScheduler([ (0,0.1),  (25,0.01),  (45,0.005)], period=50., warm_start=5) ##exp_range ##triangular2

    lrs = np.zeros((num_iters),np.float32)
    for iter in range(num_iters):
        lr = scheduler(iter)
        lrs[iter] = lr
        if lr<0:
            num_iters = iter
            break
        #print ('iter=%02d,  lr=%f   %d'%(iter,lr, scheduler.cycle))


    #plot
    fig = plt.figure()
    plot_rates(fig, lrs, title=str(scheduler))
    plt.show()


#  https://github.com/Jiaming-Liu/pytorch-lr-scheduler/blob/master/lr_scheduler.py
#  PVANET plateau lr policy
