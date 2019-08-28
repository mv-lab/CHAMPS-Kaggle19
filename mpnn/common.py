from lib.include import *
from lib.utility.draw import *
from lib.utility.file import *
from lib.net.rate import *

# ---------------------------------------------------------------------------------
COMMON_STRING = '@%s:  \n' % os.path.basename(__file__)

SEED = int(time.time())   #11 # 
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
COMMON_STRING += '\tset random seed\n'
COMMON_STRING += '\t\tSEED = %d\n' % SEED

# uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True

COMMON_STRING += '\tset cuda environment\n'
COMMON_STRING += '\t\ttorch.__version__              = %s\n' % torch.__version__
COMMON_STRING += '\t\ttorch.version.cuda             = %s\n' % torch.version.cuda
COMMON_STRING += '\t\ttorch.backends.cudnn.version() = %s\n' % torch.backends.cudnn.version()
try:
    COMMON_STRING += '\t\tos[\'CUDA_VISIBLE_DEVICES\']     = %s\n' % os.environ['CUDA_VISIBLE_DEVICES']
    NUM_CUDA_DEVICES = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
except Exception:
    COMMON_STRING += '\t\tos[\'CUDA_VISIBLE_DEVICES\']     = None\n'
    NUM_CUDA_DEVICES = 1

COMMON_STRING += '\t\ttorch.cuda.device_count()      = %d\n' % torch.cuda.device_count()
#print ('\t\ttorch.cuda.current_device()    =', torch.cuda.current_device())


COMMON_STRING += '\n'

def get_path():
    return '/run/media/windisk/Users/chrun/Documents/Projects/Predicting-Molecular-Properties/message_passing_nn/'

def get_data_path():
    return '/run/media/windisk/Users/chrun/Documents/Projects/Predicting-Molecular-Properties/data/'

# ---------------------------------------------------------------------------------
# useful : http://forums.fast.ai/t/model-visualization/12365/2


if __name__ == '__main__':
    print(COMMON_STRING)
