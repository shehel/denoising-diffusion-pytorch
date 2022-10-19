from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import torch

from clearml import Task
import pdb

task = Task.init(project_name="t4c_gen", task_name="denoising diffusion using vanilla unet")
logger = task.get_logger()
args = {
    'batch_size': 32,
    'train_lr': 1e-4,
    'save_sample_every': 5000,
    'train_steps': 700000,
    'num_workers': 16,
    'data': '../NeurIPS2021-traffic4cast/data/raw/',
    'channels': 6,
    'num_frames': 6,
    'timesteps': 100,
    'loss_type': 'l2',
    'amp': True,
    'load_model': None,
    'dim': 6,
    'cond_dim': None,
    'out_dim': 6,
    #'dim_mults': (1,2,4,8),
    'cond': False,
    'grad_accum': 2,
    'in_frames': None,
    'out_frames': None,
    'file_filter': None,
    'pad_tuple': (6,6,1,0),
    'h': 495,
    'w': 436,
    }

task.connect(args)
print ('Arguments: {}'.format(args))

model = Unet(
    dim = args['dim'],
    cond_dim =args['cond_dim'],
    out_dim = args["out_dim"]
)
im_size = (args['h']+args['pad_tuple'][-1]+args['pad_tuple'][-2],
           (args['w']+args['pad_tuple'][-3]+args['pad_tuple'][-4]))

model = torch.nn.DataParallel(model)
diffusion = GaussianDiffusion(
    model,
    image_size = im_size,
    timesteps = args['timesteps'],   # number of steps
    loss_type = args['loss_type'],    # L1 or L2
    ).cuda()

trainer = Trainer(
    diffusion,
    args['data'],                         # this folder path needs to contain all your training data, as .gif files, of correct image size and number of frames
    train_batch_size = args['batch_size'],
    train_lr = args['train_lr'],
    save_and_sample_every = args['save_sample_every'],
    train_num_steps = args['train_steps'],         # total training steps
    gradient_accumulate_every = args['grad_accum'],    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = args['amp'],
    im_size = im_size,
    cond = args['cond'],
    in_frames = args['in_frames'],
    out_frames = args['out_frames'],
    file_filter = args['file_filter'],
    pad_tuple = args['pad_tuple'],
    h = args['h'],
    w = args['w'],
    num_workers = args['num_workers'],
)

if args['load_model'] is not None:
    trainer.load(args['load_model'])

trainer.train(logger=logger)
#trainer.infer(logger=logger, milestone=args['load_model'])
