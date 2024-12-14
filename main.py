
import argparse
import time
import torch
import numpy as np
import torch.optim as optim
import csv
from skimage.io import imread, imshow
# custom modules

from utils import get_model, to_device, prepare_dataloader, transform2Dto3D
from loss import TwinLoss
import os
import torch.nn.functional as F
import PIL.Image as pil
from torchvision import transforms
import cv2
import matplotlib.cm as cm
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from torch.utils.data import Subset

# plot params

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (15, 10)

file_dir = os.path.dirname(__file__)  # the directory that main.py resides in


def return_arguments():
    parser = argparse.ArgumentParser(description='PyTorch ')

    parser.add_argument('--data_dir',
                        type=str,
                        help='path to the dataset folder',
                        default='./coffbea-2023/data')
    parser.add_argument('--model_path',
                        default=os.path.join(file_dir, "weights"),
                        help='path to the trained model')
    parser.add_argument('--output_directory',
                        default=os.path.join(file_dir, "output"),
                        help='output directory')
    parser.add_argument('--is_stereo', default=True,
                        help='input stereo images')
    parser.add_argument('--input_height', type=int, help='input height',
                        default=306)
    parser.add_argument('--input_width', type=int, help='input width',
                        default=230)
    parser.add_argument('--full_height', type=int, help='input height',
                        default=920)
    parser.add_argument('--full_width', type=int, help='input width',
                        default=1224)
    parser.add_argument('--size', type=tuple, help='input size',
                        default=(1224, 1224))  # (920, 1224) default=(896, 896)
    parser.add_argument('--model', default='stereo_res_mlp',
                        help='mono_res, mono_vit, mono_vit_mlp, mono_res_mlp, '
                             'stereo_res, stereo_vit, stereo_vit_mlp, stereo_res_mlp, '
                             'stereo_res_lstm, stereo_vit_lstm, stereo_mask_res_mlp, stereo_img_mask_res_mlp, '
                             'stereo_img_contour_res_mlp, stereo_mask_contour_res_mlp, stereo_rgbm_res_mlp',)
    parser.add_argument('--resume', default=None,
                        help='load weights to continue train from where it last stopped')
    parser.add_argument('--load_weights_folder', default=os.path.join(file_dir, "weights"),
                        help='folder to load weights to continue train from where it last stopped')
    parser.add_argument('--pretrained', default=True,
                        help='Use weights of pretrained model')
    parser.add_argument('--mode_flag', default='train',
                        help='mode: train or test (default: train)')
    parser.add_argument('--epochs', type=int, default=2,
                        help='number of total epochs to run')
    parser.add_argument('--startEpoch', type=int, default=0,
                        help='number of total epochs to run')
    parser.add_argument('--testepoch', type=str, default='border_cpt',
                        help='number of total epochs to test')
    parser.add_argument('--learning_rate', default=1e-4,
                        help='initial learning rate (default: 1e-4)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--adjust_lr', default=True,
                        help='apply learning rate decay or not\
                        (default: True)')
    parser.add_argument('--device',
                        default='cuda:0',
                        help='choose cpu or cuda:0 device"')
    parser.add_argument('--do_augmentation', default=True,
                        help='do augmentation of images or not')
    parser.add_argument('--augment_parameters', default=[0.8, 1.2, 0.5, 2.0, 0.8, 1.2],
                        help='lowest and highest values for gamma,\
                        brightness and color respectively')
    parser.add_argument('--print_weights', default=False,
                        help='print weights of every layer')
    parser.add_argument('--input_channels', default=6,
                        help='Number of channels in input tensor')
    parser.add_argument('--num_workers', default=1,
                        help='Number of workers in dataloader')
    parser.add_argument('--use_multiple_gpu', default=True)
    parser.add_argument('--debug_mode', default=False,
                        help='Whether use the debug mode (partial dataset)')
    parser.add_argument('--debug_train_size', default=60,
                        help='The training size used for debugging')
    parser.add_argument('--debug_val_size', default=20,
                        help='The validation size used for debugging')

    args = parser.parse_args()
    return args


def adjust_learning_rate(optimizer, epoch, learning_rate):
    if epoch >= 30 and epoch < 60:
        lr = learning_rate / 2
    elif epoch >= 60 and epoch < 80:
        lr = learning_rate / 4
    elif epoch >= 80:
        lr = learning_rate / 8
    else:
        lr = learning_rate
    # if epoch >= 300 and epoch < 400:
    #     lr = learning_rate / 2
    # elif epoch >= 400:
    #     lr = learning_rate / 4
    # else:
    #     lr = learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Model:
    def __init__(self, args):
        self.args = args
        # create weight folder
        if os.path.isdir(args.model_path):
            print('Weights folder exists')
        else:
            print('Weights folder create')
            os.makedirs(args.model_path)

        # Set up model
        self.device = args.device
        self.model = get_model(args.model, pretrained=args.pretrained)
        self.model = self.model.to(self.device)
        if args.use_multiple_gpu:
            self.model = torch.nn.DataParallel(self.model)

        # resume
        self.best_val_loss = float('Inf')

        # lists for storing loss history
        self.training_loss_history = []

        self.validation_loss_history = []

        if args.mode_flag == 'train':
            self.loss_function = TwinLoss(
                L1_w=0.5,
                L2_w=0.5).to(self.device)

            self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)

            if args.resume is not None:
                self.load_model_continue_train(os.path.join(self.args.model_path, 'weights_cpt.pt'))  # default: weights_last.pt
                self.args.startEpoch = self.startEpoch
                self.best_val_loss = self.best_val_loss

            # load val data first
            self.val_n_img, self.val_loader = prepare_dataloader(data_directory=args.data_dir,
                                                                 is_stereo=args.is_stereo,
                                                                 batch_size=args.batch_size,
                                                                 num_workers=args.num_workers,
                                                                 size=args.size,
                                                                 mode_flag='val',
                                                                 debug_mode=args.debug_mode,
                                                                 debug_size=args.debug_val_size)
            # Load train data
            self.train_n_img, self.train_loader = prepare_dataloader(data_directory=args.data_dir,
                                                                     is_stereo=args.is_stereo,
                                                                     batch_size=args.batch_size,
                                                                     num_workers=args.num_workers,
                                                                     size=args.size,
                                                                     mode_flag='train',
                                                                     debug_mode=args.debug_mode,
                                                                     debug_size=args.debug_train_size)
        else:
            args.test_model_path = os.path.join(self.args.model_path, args.testepoch + '.pth')
            self.model.load_state_dict(torch.load(args.test_model_path))
            args.batch_size = 1
            self.test_n_img, self.test_loader = prepare_dataloader(data_directory=args.data_dir,
                                                                   is_stereo=args.is_stereo,
                                                                   batch_size=args.batch_size,
                                                                   num_workers=args.num_workers,
                                                                   size=args.size,
                                                                   mode_flag='test',
                                                                   debug_mode=False)

        if 'cuda' in self.device:
            torch.cuda.synchronize()

    def train(self):
        val_losses = []
        running_val_loss = 0.0
        self.model.eval()   # Before training, validate current performance
        for data in self.val_loader:
            data = to_device(data, self.device)  # dict
            images = data['images']
            #print(images.size())
            points = torch.flatten(data['points'], 1)
            left_gt = data['labels']
            depth_map = data['depth_map']

            # left_pred = self.model(images)
            # left_pred = self.model(images, points)
            # left_pred = self.model(images, depth_map)
            left_pred = self.model(images, points, depth_map)

            loss= self.loss_function(left_pred, left_gt)
            val_losses.append(loss.item())
            running_val_loss += loss.item()

        running_val_loss /= self.val_n_img / self.args.batch_size

        """ record the loss """
        with open('losses.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Training Loss', 'Validation Loss', 'Training Base Loss', 'Training Att Loss',
                             'Validation Base Loss', 'Validation Att Loss'])
            for epoch in range(self.args.startEpoch, self.args.epochs):  # Training
                if self.args.adjust_lr:
                    adjust_learning_rate(self.optimizer, epoch,
                                         self.args.learning_rate)
                c_time = time.time()
                running_loss = 0.0
                training_losses = []
                self.model.train()
                for data in self.train_loader:
                    # Load data
                    data = to_device(data, self.device)
                    images = data['images']
                    points = torch.flatten(data['points'], 1)
                    left_gt = data['labels']
                    depth_map = data['depth_map']
                    # One optimization iteration
                    self.optimizer.zero_grad()  # Clear the gradient
                    # left_pred = self.model(images)
                    # left_pred = self.model(images, points)
                    left_pred = self.model(images, depth_map)
                    # left_pred = self.model(images, points, depth_map)

                    # loss = self.loss_function(left_pred, left_gt)   # Calculate the loss
                    loss= self.loss_function(left_pred, left_gt) # 无attention
                    loss.backward() # Backpropagation, calculate gradients for every parameter
                    self.optimizer.step()   # Update the parameters
                    training_losses.append(loss.item()) # Append losses for this epoch
                    running_loss += loss.item()  # Calculate the total losses

                running_val_loss = 0.0
                self.model.eval()   # Evaluate the model after training - xsy
                for data in self.val_loader:
                    data = to_device(data, self.device)
                    images = data['images']
                    points = torch.flatten(data['points'], 1)
                    left_gt = data['labels']
                    depth_map = data['depth_map']

                    # left_pred = self.model(images)
                    # left_pred = self.model(images, points)
                    # left_pred = self.model(images, depth_map)
                    left_pred = self.model(images, points, depth_map)

                    loss= self.loss_function(left_pred, left_gt)

                    val_losses.append(loss.item())
                    running_val_loss += loss.item()

                # Estimate loss per image
                running_loss /= self.train_n_img / self.args.batch_size
                running_val_loss /= self.val_n_img / self.args.batch_size

                print(
                    'Epoch:',
                    epoch + 1,
                    'train_loss:',
                    running_loss,
                    'val_loss:',
                    running_val_loss,
                    'time:',
                    round(time.time() - c_time, 3),
                    's',
                )

                self.training_loss_history.append(running_loss)
                self.validation_loss_history.append(running_val_loss)

                writer.writerow([
                    epoch + 1,
                    running_loss,
                    running_val_loss,
                ])

                # save weights for every 50 epoch
                if epoch % 50 == 0:
                    self.save(os.path.join(self.args.model_path, 'epoch{}.pth'.format(str(epoch))))
                # if running_val_loss < best_val_loss:
                if running_val_loss < self.best_val_loss:
                    self.save(os.path.join(self.args.model_path, 'border_cpt.pth'))
                    self.best_val_loss = running_val_loss
                    self.save_continue_train(epoch, running_val_loss, 'weights_cpt.pt')
                    print('Model_saved')

                self.save(os.path.join(self.args.model_path, 'border_last.pth'))
                self.save_continue_train(epoch, running_loss, 'weights_last.pt')

        print('Finished Training.')
        # self.save(os.path.join(self.args.model_path, 'train_end.pth'))
        self.save_continue_train(self.args.epochs, running_val_loss, 'train_end.pt')

        self.plot_losses()
        print("Loss plotted")

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def save_continue_train(self, epoch, loss, path):
        save_path = os.path.join(self.args.model_path, path)
        torch.save({'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss,
                    'best_val_loss': self.best_val_loss
                    }, save_path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def load_model_continue_train(self, path):
        assert os.path.isfile(path), \
            "Cannot find folder {}".format(path)
        print("loading model from folder {}".format(path))
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.startEpoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']


    def error3d_inference(self, args):
        self.model.eval()
        self.args.output_directory = os.path.join(self.args.output_directory, 'all_on')
        error3d = []
        error2d = []
        r2pred = []
        r2gt = []

        with open('results.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['index', '2d_pred_x', '2d_pred_y', '2d_gt_x', '2d_gt_y', '2d_error',
                             '3d_pred_x', '3d_pred_y', '3d_pred_z', '3d_gt_x', '3d_gt_y', '3d_gt_z', '3d_error'])

            index = 0
            for data in self.test_loader:
                data = to_device(data, self.device)
                images = data['images']
                points = torch.flatten(data['points'], 1)
                left_gt = data['labels']
                depthgt = data['depthgt'].cpu().detach().numpy().squeeze(0)
                depth_map = data['depth_map']
                padding_left = data['padding_left'].cpu().numpy()
                padding_top = data['padding_top'].cpu().numpy()

                left_pred = self.model(images, points).cpu().detach().numpy()
                left_gt = left_gt.cpu().detach().numpy()

                # recover from padded tensor
                left_pred[:, 0] -= padding_left
                left_pred[:, 1] -= padding_top
                left_gt[:, 0] -= padding_left
                left_gt[:, 1] -= padding_top

                # error 2d
                loss2d = np.linalg.norm(left_pred - left_gt)
                error2d.append(loss2d)

                # error 3d
                if (int(left_pred[0, 0]) < args.full_width) and (int(left_pred[0, 1]) < args.full_height):

                    Z_pred = depthgt[int(left_pred[0, 1]), int(left_pred[0, 0])]
                    Z_gt = depthgt[int(left_gt[0, 1]), int(left_gt[0, 0])]
                    interval = 5
                    if Z_pred == 0.0:
                        Z_pred = np.median(depthgt[(int(left_pred[0, 1]) - interval):(int(left_pred[0, 1]) + interval),
                                           (int(left_pred[0, 0]) - interval):(int(left_pred[0, 0]) + interval)])
                    if Z_gt == 0.0:
                        Z_gt = np.median(depthgt[(int(left_gt[0, 1]) - interval):(int(left_gt[0, 1]) + interval),
                                         (int(left_gt[0, 0]) - interval):(int(left_gt[0, 0]) + interval)])

                    if (Z_gt != 0.0) and (Z_pred != 0.0):
                        pred_3d = transform2Dto3D(Z_pred, left_pred, alpha=2894.8, beta=2902.2, ox=612, oy=460)
                        gt_3d = transform2Dto3D(Z_gt, left_gt, alpha=2894.8, beta=2902.2, ox=612, oy=460)
                        loss3d = np.linalg.norm(pred_3d - gt_3d)
                        error3d.append(loss3d)

                        writer.writerow([index, left_pred[0, 0], left_pred[0, 1], left_gt[0, 0], left_gt[0, 1], loss2d,
                                         pred_3d[0], pred_3d[1], pred_3d[2], gt_3d[0], gt_3d[1], gt_3d[2], loss3d])
                    else:
                        writer.writerow(
                            [index, left_pred[0, 0], left_pred[0, 1], left_gt[0, 0], left_gt[0, 1], loss2d] + ['NA'] * 6)
                else:
                    writer.writerow(
                        [index, left_pred[0, 0], left_pred[0, 1], left_gt[0, 0], left_gt[0, 1], loss2d] + ['NA'] * 6)

                # r2 square
                r2pred.append(left_pred[0])
                r2gt.append(left_gt[0])

                index += 1

        print(len(error3d))
        mean = np.mean(error3d)
        std = np.std(error3d)
        median = np.median(error3d)
        print("\n  " + ("{:>8} | " * 3).format("3d_mean", "3d_std", "3d_median"))
        print(("&{: 8.1f}  " * 3).format(mean, std, median) + "\\\\")
        print("\n-> error3d Done!")

        mean = np.mean(error2d)
        std = np.std(error2d)
        median = np.median(error2d)
        print("\n  " + ("{:>8} | " * 3).format("2d_mean", "2d_std", "2d_median"))
        print(("&{: 8.1f}  " * 3).format(mean, std, median) + "\\\\")
        print("\n-> error2d Done!")

        r_square = r2_score(r2pred, r2gt)
        print('r_square = ', r_square)


    def error3d_direct(self, args):
        self.model.eval()
        self.args.output_directory = os.path.join(self.args.output_directory, 'all_on')
        error3d = []
        error2d = []
        r2pred = []
        r2gt = []

        with open('results_3d.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['index', '3d_pred_x', '3d_pred_y', '3d_pred_z', '3d_gt_x', '3d_gt_y', '3d_gt_z', '3d_error'])
            index = 0
            for data in self.test_loader:
                data = to_device(data, self.device)
                images = data['images']
                points = torch.flatten(data['points'], 1)
                left_gt = data['labels']
                depthgt = data['depthgt'].cpu().detach().numpy().squeeze(0)
                depth_map = data['depth_map']

                padding_left = data['padding_left'].cpu().numpy()
                padding_top = data['padding_top'].cpu().numpy()

                left_pred = self.model(images).cpu().detach().numpy()
                left_gt = left_gt.cpu().detach().numpy()

                # recover from padded tensor
                left_pred[:, 0] -= padding_left
                left_pred[:, 1] -= padding_top
                left_gt[:, 0] -= padding_left
                left_gt[:, 1] -= padding_top

                # error 3d
                loss3d = np.linalg.norm(left_pred - left_gt)
                error3d.append(loss3d)

                # r2 square
                r2pred.append(left_pred[0])
                r2gt.append(left_gt[0])

                writer.writerow([index, left_pred[0, 0], left_pred[0, 1], left_pred[0, 2], \
                                 left_gt[0, 0], left_gt[0, 1], left_gt[0, 2], loss3d])

                index += 1

        print(len(error3d))
        mean = np.mean(error3d)
        std = np.std(error3d)
        median = np.median(error3d)
        print("\n  " + ("{:>8} | " * 3).format("3d_mean", "3d_std", "3d_median"))
        print(("&{: 8.1f}  " * 3).format(mean, std, median) + "\\\\")
        print("\n-> error3d Done!")

        r_square = r2_score(r2pred, r2gt)
        print('r_square = ', r_square)

    def plot_losses(self):

        plt.figure(figsize=(10, 5))
        plt.plot(self.training_loss_history, label='Training Loss')
        plt.plot(self.validation_loss_history, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig('loss.jpg')

def main():
    args = return_arguments()

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    num_devices = torch.cuda.device_count()
    print(f"Number of CUDA devices available: {num_devices}")

    if args.mode_flag == 'train':
        model = Model(args)
        model.train()
    elif args.mode_flag == 'test':
        model_test = Model(args)
        # model_test.test()
        model_test.error3d_inference(args)


if __name__ == '__main__':
    main()
    # os.system("/usr/bin/shutdown")




