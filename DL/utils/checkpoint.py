import os
import torch


def save_checkpoint(state, is_best, epoch, save_path='./', save_image_model=False, stage=0):
    print("=> saving checkpoint '{}'".format(epoch))
    if(epoch % 20 == 0):
        if save_image_model:
            torch.save(state, os.path.join(
                save_path, 'image_feature_net_checkpoint_%03d.pth.tar' % epoch))
        else:
            torch.save(state, os.path.join(
                save_path, 'checkpoint_%03d.pth.tar' % epoch))
    if is_best:
        if save_image_model:
            torch.save(state, os.path.join(save_path, 'image_feature_net_model_best_checkpoint.pth.tar'))
        else:
            if not stage:
                torch.save(state, os.path.join(save_path, 'model_best_checkpoint.pth.tar'))
            else:
                torch.save(state, os.path.join(save_path, f'model_best_checkpoint_stage{stage}.pth.tar'))


def load_checkpoint(args, model, optimizer=None, verbose=True):

    checkpoint = torch.load(args.resume)

    start_epoch = 0
    best_mse = float('inf')

    if "epoch" in checkpoint:
        start_epoch = checkpoint['epoch']

    if "best_mse" in checkpoint:
        best_mse = checkpoint['best_mse']

    model.load_state_dict(checkpoint['state_dict'], False)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(args.device)

    if verbose:
        print("=> loading checkpoint '{}' (epoch {})"
              .format(args.resume, start_epoch))

    return model, optimizer, best_mse, start_epoch