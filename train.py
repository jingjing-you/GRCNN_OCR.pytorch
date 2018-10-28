#coding:utf-8
import numpy as np
import torch
import GRCNN
import dataloader
from torch.autograd import Variable
import torch.optim as optim
import cv2
import os
from warpctc_pytorch import CTCLoss
from glob import glob
import utils
import shutil
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='PyTorch GCRNN Training')

parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.5, type=float, help='beta1')
parser.add_argument('--beta2', default=0.999, type=float, help='beta2')
parser.add_argument('--epoches', default=2, type=int, help='the number of train')
parser.add_argument('--is_use_gpu', default=False, type=bool, help='is use gpu')
parser.add_argument('--gpu_list', default='-1', type=str, help='gpu_list')
parser.add_argument('--batch_size', default=2, type=int, help='min batch size')
parser.add_argument('--n_class', default=37, type=int, help='the number of class')
parser.add_argument('--n_hidden', default=64, type=int, help='the number of hidden unit')
parser.add_argument('--img_w', default=100, type=int, help='the width of image')
parser.add_argument('--img_h', default=32, type=int, help='the height of image')
parser.add_argument('--max_len', default=20, type=int, help='max length(for data load)')
parser.add_argument('--num_worker', default=2, type=int, help='the number of work')
parser.add_argument('--alphabet', default='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', type=str, help='alphabet')
parser.add_argument('--model_save_path', default='./model_result/', type=str, help='model_save_path')
parser.add_argument('--train_main_path', default='./data_sample/', type=str, help='train data path')
parser.add_argument('--val_main_path', default='./data_sample/', type=str, help='validation set path')


args = parser.parse_args()


if args.is_use_gpu:
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu_list

if os.path.exists(args.model_save_path):
    shutil.rmtree(args.model_save_path)
os.mkdir(args.model_save_path)

dataset = dataloader.MyDataSet(args.train_main_path, width=args.img_w, height=args.img_h, max_len = args.max_len)
trainloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker)
val_img_list = glob(args.val_main_path + '*.jpg')

# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
crnn = GRCNN.GRCNN(args.n_class)
crnn.apply(weights_init)


if args.is_use_gpu:
    crnn = crnn.cuda()
    criterion = CTCLoss().cuda()
else:
    criterion = CTCLoss()
#net.cuda()
print('net has load!')
converter = utils.strLabelConverter(args.alphabet)

optimizer=optim.Adam(crnn.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

best_acc=-1
totalLoss=[]
avg_test_acc = []
avg_train_acc = []

def get_img(img_path):
    img = cv2.imread(img_path)
    img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (args.img_w, args.img_h))
    img = np.reshape(img, newshape=[1,1, args.img_h, args.img_w])
    img = img.astype(np.float32)
    img = img / 255
    img = img - 0.5
    img = img * 2
    img_tensor = torch.from_numpy(img).float()
    true_label = img_path.replace('.jpg', '').split('_')[-1]
    return img_tensor, true_label


for epoch in range(1, args.epoches, 1):
    #train
    for p in crnn.parameters():
        p.requires_grad = True
    crnn.train()
    avg_totalLoss=0.0
    train_acc = 0.0
    test_acc = 0.0
    for batch_id,(img_tensor, txt_len, txt_label, txt_name) in tqdm(enumerate(trainloader)):
        optimizer.zero_grad()
        batch_length = img_tensor.size(0)
        txt_label = txt_label.numpy().reshape(args.max_len*batch_length)
        txt_label = torch.from_numpy(np.array([item for item in txt_label if item != 0]).astype(np.int))
        if args.is_use_gpu:
            img_tensor = Variable(img_tensor.float()).cuda()
        else:
            img_tensor = Variable(img_tensor.float())
        txt_len = Variable(txt_len.int()).squeeze(1)
        txt_label = Variable(txt_label.int())

        preds = crnn(img_tensor)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_length))
        total_loss = criterion(preds, txt_label, preds_size, txt_len) / batch_length
        total_loss.backward()
        optimizer.step()
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, txt_name):
            if pred == target:
                train_acc += 1
        avg_totalLoss += total_loss.item()
        info='epoch : %d ,process: %d/%d ,  totalLoss: %f , lr: %f  ' % (epoch, batch_id, trainloader.__len__(), total_loss.item(), optimizer.param_groups[0]['lr'])
        #print(info)
        #break
    train_acc /= len(dataset)
    avg_train_acc.append(train_acc)

    #valid
    for p in crnn.parameters():
        p.requires_grad = False
    crnn.eval()
    for img_path in val_img_list:
        img_tensor, gt = get_img(img_path)
        if args.is_use_gpu:
            img_tensor = Variable(img_tensor).cuda()
        else:
            img_tensor = Variable(img_tensor)
        preds = crnn(img_tensor)
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
        sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

        print('%-33s => %-33s, gt: %-20s, %s' % (raw_pred, sim_pred, gt, sim_pred == gt))
        if sim_pred == gt:
            test_acc += 1
        #break

    test_acc /= len(val_img_list)
    avg_test_acc.append(test_acc)
    avg_totalLoss /= trainloader.__len__()
    totalLoss.append(avg_totalLoss)

    info='epoch : %d , avg_totalLoss: %f , lr: %f  avg_test_acc: %f   avg_train_acc: %f ' % (epoch, avg_totalLoss, optimizer.param_groups[0]['lr'], test_acc, train_acc)
    print(info)
    # save result
    if best_acc <= test_acc:
        best_acc = test_acc
        if args.is_use_gpu:
            torch.save(crnn.cpu().state_dict(), args.model_save_path + 'cpu_model_parameter_' + str(epoch) + '.pkl')
            crnn.cuda()
        else:
            torch.save(crnn.state_dict(), args.model_save_path + 'cpu_model_parameter_' + str(epoch) + '.pkl')
        np.savetxt(args.model_save_path + 'total_loss.csv', totalLoss)
        np.savetxt(args.model_save_path + 'avg_test_acc.csv', avg_test_acc)
        np.savetxt(args.model_save_path + 'avg_train_acc.csv', avg_train_acc)

    if test_acc == 1:
        break
    if(epoch % 50 == 0):
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.6




