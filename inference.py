import os, torch, argparse, tqdm, random, numpy as np
from util.dataset import create_data
from util.noiser import create_noiser
from util.model import create_model
from torchvision.utils import save_image
from diffusers import AutoencoderKL

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def match_ckpt(ckpt):
    _ckpt = {}
    for k, v in ckpt.items():
        if 'module.' in k:
            k = k.replace('network.module.', 'network.')
        _ckpt[k] = v
    return _ckpt

def main(args):
    seed_everything(42)

    device = torch.device(f'cuda')

    prior_set, _, _ = create_data(args.prior, 1, dataset_size=args.num_sample, batch_size=args.batch_size)
    data_set, _, _ = create_data(args.dataset, 1, dataset_size=args.num_sample, batch_size=args.batch_size)

    noiser = create_noiser(args.noiser, args, device)

    forward_model = create_model(args.method, args, device, noiser, rank=0, direction='f')
    backward_model = create_model(args.method, args, device, noiser, rank=0, direction='b')

    forward_model.to(device)
    backward_model.to(device)

    ckpt = torch.load(args.ckpt, map_location='cpu')
    forward_model.load_state_dict(match_ckpt(ckpt['forward_model']), strict=True)
    backward_model.load_state_dict(match_ckpt(ckpt['backward_model']), strict=True)

    with torch.no_grad():

        forward_model.eval()
        backward_model.eval()

        for direction in ['p', 'q']:

            save_path = os.path.join('inference', args.exp_name, f'{direction}')
            os.makedirs(os.path.join(save_path, 'batch'), exist_ok=True)
            os.makedirs(os.path.join(save_path, 'single'), exist_ok=True)

            for i in tqdm.trange(args.num_sample // args.batch_size, desc=f'{direction} inference'):

                if direction == 'p':
                    x_prior = torch.stack(
                        [prior_set[_i + i * args.batch_size] for _i in range(args.batch_size)], dim=0
                    ).to(device)
                    ps = forward_model.inference(x_prior, sample=True)[0]
                    samples = ps[-1]
                elif direction == 'q':
                    x_data = torch.stack(
                        [data_set[_i + i * args.batch_size] for _i in range(args.batch_size)], dim=0
                    ).to(device)
                    qs = backward_model.inference(x_data, sample=True)[0]
                    samples = qs[-1]

                if samples.shape[1] == 4:
                    vae = AutoencoderKL.from_pretrained(
                        "stabilityai/stable-diffusion-xl-base-1.0", subfolder="vae", revision=None, variant=None
                    )
                    vae = vae.to(device)
                    samples = vae.decode(samples.to(device) / vae.config.scaling_factor)["sample"].cpu().clamp(-1, 1)

                save_image(samples, os.path.join(save_path, 'batch', f'{i:05d}.png'), nrow=4, normalize=True, value_range=(-1, 1))
                for j in range(args.batch_size):
                    save_image(samples[j], os.path.join(save_path, 'single', f'{i*args.batch_size+j:05d}.png'), normalize=True, value_range=(-1, 1))

def create_parser():
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--num_sample', type=int, default=128, help='number of samples')
    argparser.add_argument('--batch_size', type=int, default=16, help='batch size')
    
    argparser.add_argument('--method', type=str, default='dsb', help='method')
    argparser.add_argument('--simplify', action='store_true', help='whether to use simplified DSB')
    argparser.add_argument('--reparam', type=str, default=None, help='whether to use reparameterized DSB, "term" for TR-DSB, "flow" for FR-DSB')
    argparser.add_argument('--noiser', type=str, default='flow', help='noiser type, "flow" noiser for Flow Matching models, "dsb" noiser for DSB models')
    argparser.add_argument('--gamma_type', type=str, default='constant', help='gamma schedule for DSB')
    argparser.add_argument('--training_timesteps', type=int, default=100, help='training timesteps')
    argparser.add_argument('--inference_timesteps', type=int, default=100, help='inference timesteps')
    
    argparser.add_argument('--network', type=str, default='mlp', help='network architecture to use')
    argparser.add_argument('--use_amp', action='store_true', help='whether to use mixed-precision training')

    argparser.add_argument('--prior', type=str, default='standard', help='prior distribution')
    argparser.add_argument('--dataset', type=str, default='checkerboard:4', help='data distribution')

    argparser.add_argument('--exp_name', type=str, default='try', help='name of experiment')
    argparser.add_argument('--ckpt', type=str, default=None, help='checkpoint to load')

    return argparser


if __name__ == '__main__':

    argparser = create_parser()
    args = argparser.parse_args()

    if 'dsb' in args.method:
        assert args.training_timesteps == args.inference_timesteps

    main(args)
