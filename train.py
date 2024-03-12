import argparse
import os
import sys
import torch
from torch.backends import cudnn
sys.path.append('.')
from Config import *
from datas import make_data_loader
from modeling import build_model
from solver import make_optimizer, make_optimizer_with_center, WarmupMultiStepLR
from losses import make_loss, make_loss_with_center
from trainer import do_train, do_train_with_center
from logger import setup_logger
MODEL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train():
    # prepare dataset
    train_loader, val_loader, num_query, num_classes = make_data_loader()

    # prepare model
    model = build_model(num_classes)

    if MODEL_IF_WITH_CENTER == 'no':
        print('Train without center loss, the loss type is', MODEL_METRIC_LOSS_TYPE)
        optimizer = make_optimizer(model)
        # scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
        #                               cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

        loss_func = make_loss(num_classes)     # modified by gu

        # Add for using self trained model
        if MODEL_PRETRAIN_CHOICE == 'self':
            start_epoch = eval(MODEL_PRETRAIN_PATH.split('/')[-1].split('.')[0].split('_')[-1])
            print('Start epoch:', start_epoch)
            path_to_optimizer = MODEL_PRETRAIN_PATH.replace('model', 'optimizer')
            print('Path to the checkpoint of optimizer:', path_to_optimizer)
            model.load_state_dict(torch.load(MODEL_PRETRAIN_PATH))
            optimizer.load_state_dict(torch.load(path_to_optimizer))
            scheduler = WarmupMultiStepLR(optimizer, SOLVER_STEPS, 
                                          SOLVER_GAMMA, SOLVER_WARMUP_FACTOR,
                                          SOLVER_WARMUP_ITERS, SOLVER_WARMUP_METHOD, start_epoch)
        elif MODEL_PRETRAIN_CHOICE == 'imagenet':
            start_epoch = 0
            scheduler = WarmupMultiStepLR(optimizer, SOLVER_STEPS, 
                                          SOLVER_GAMMA, SOLVER_WARMUP_FACTOR,
                                          SOLVER_WARMUP_ITERS, SOLVER_WARMUP_METHOD)
        else:
            print('Only support pretrain_choice for imagenet and self, but got {}'.format(
                MODEL_PRETRAIN_CHOICE))

        arguments = {}

        do_train(
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,      # modify for using self trained model
            loss_func,
            num_query,
            start_epoch     # add for using self trained model
        )
    elif MODEL_IF_WITH_CENTER == 'yes':
        print('Train with center loss, the loss type is', MODEL_METRIC_LOSS_TYPE)
        loss_func, center_criterion = make_loss_with_center(num_classes)  # modified by gu
        optimizer, optimizer_center = make_optimizer_with_center(model, center_criterion)
        # scheduler = WarmupMultiStepLR(optimizer, SOLVER_STEPS, 
#                                         SOLVER_GAMMA, SOLVER_WARMUP_FACTOR,
        #                               SOLVER_WARMUP_ITERS, SOLVER_WARMUP_METHOD)

        arguments = {}

        # Add for using self trained model
        if MODEL_PRETRAIN_CHOICE == 'self':
            start_epoch = eval(MODEL_PRETRAIN_PATH.split('/')[-1].split('.')[0].split('_')[-1])
            print('Start epoch:', start_epoch)
            path_to_optimizer = MODEL_PRETRAIN_PATH.replace('model', 'optimizer')
            print('Path to the checkpoint of optimizer:', path_to_optimizer)
            path_to_center_param = MODEL_PRETRAIN_PATH.replace('model', 'center_param')
            print('Path to the checkpoint of center_param:', path_to_center_param)
            path_to_optimizer_center = MODEL_PRETRAIN_PATH.replace('model', 'optimizer_center')
            print('Path to the checkpoint of optimizer_center:', path_to_optimizer_center)
            model.load_state_dict(torch.load(MODEL_PRETRAIN_PATH))
            optimizer.load_state_dict(torch.load(path_to_optimizer))
            center_criterion.load_state_dict(torch.load(path_to_center_param))
            optimizer_center.load_state_dict(torch.load(path_to_optimizer_center))
            scheduler = WarmupMultiStepLR(optimizer, SOLVER_STEPS, 
                                          SOLVER_GAMMA, SOLVER_WARMUP_FACTOR,
                                          SOLVER_WARMUP_ITERS, SOLVER_WARMUP_METHOD, start_epoch)
        elif MODEL_PRETRAIN_CHOICE == 'imagenet':
            start_epoch = 0
            scheduler = WarmupMultiStepLR(optimizer, SOLVER_STEPS, 
                                          SOLVER_GAMMA, SOLVER_WARMUP_FACTOR,
                                          SOLVER_WARMUP_ITERS, SOLVER_WARMUP_METHOD)
        else:
            print('Only support pretrain_choice for imagenet and self, but got {}'.format(
                MODEL_PRETRAIN_CHOICE))

        do_train_with_center(
            model,
            center_criterion,
            train_loader,
            val_loader,
            optimizer,
            optimizer_center,
            scheduler,      # modify for using self trained model
            loss_func,
            num_query,
            start_epoch     # add for using self trained model
        )
    else:
        print("Unsupported value for cfg.MODEL.IF_WITH_CENTER {}, only support yes or no!\n".format(
            MODEL_IF_WITH_CENTER))


def main():
#     parser = argparse.ArgumentParser(description="ReID Baseline Training")
#     parser.add_argument(
#         "--config_file", default="", help="path to config file", type=str
#     )
#     parser.add_argument("opts", help="Modify config options using the command-line", default=None,
#                         nargs=argparse.REMAINDER)

#     args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

#     if args.config_file != "":
#         merge_from_file(args.config_file)
#     merge_from_list(args.opts)
#     freeze()

    output_dir = OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
#     logger.info(args)

#     if args.config_file != "":
#         logger.info("Loaded configuration file {}".format(args.config_file))
#         with open(args.config_file, 'r') as cf:
#             config_str = "\n" + cf.read()
#             logger.info(config_str)
#     logger.info("Running with config:\n{}".format())

    if MODEL_DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = MODEL_DEVICE_ID    # new add by gu
    cudnn.benchmark = True
    train()


if __name__ == '__main__':
    main()