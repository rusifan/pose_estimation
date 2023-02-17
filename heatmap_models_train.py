import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
from tqdm import tqdm
from model.resnet import resnet50, resnet10, resnet18
from model.stackedhour import hg2
import numpy as np
from dataloader.heatmap import hum36m_dataloader
# from dataloader.loader_hog import hum36m_dataloader
from utils.loss import joints_mse_loss, accuracy
from utils.utils1 import *
from utils.eval_cal import mpjpe


num_layers = 2
annotation_path_train = "annotation_body3d/fps25/h36m_train.npz"
annotation_path_test = "annotation_body3d/fps25/h36m_test.npz"
root_data_path = "/netscratch/nafis/human-pose/dataset_human36_nos7_f25"
batchSize_train = 96
batchSize_test = 16 #256 if using 168G mem 576756 780040
workers = 6
wandb_flag = True
# run_name = "res_gcn_2mgcn_fixLoss_Fps25_noS7"
#hog loader is used chamge it when needed
run_name = "stacked_parallel"
model_name = 'stacked_high_batch'
lr = 5e-3 #learning rate 0.005
# lr = 5e-2 #learning rate 
optimizer_name = 'Adam'
model_reload = 0
epoch_train = 30
train_flag = 1
test_flag = 1
best_error = math.inf
previous_module_gcn_name = ""
save_dir = "./results/stacked_dataparallel"
save_out_type = "xyz"
large_decay_epoch = 4
lr_decay = 0.90
save_model_flag = 1
######################################################
torch.manual_seed(0)


adj = np.load('/netscratch/nafis/human-pose/real_time_pose/model/adj_4.npy')
adj = torch.from_numpy(adj).to('cuda')
# model = MyNet(adj=adj, block=num_layers).to('cuda')
# model = resnet18(pretrained=False, num_classes=17*2).to('cuda')
# model = MyNet(pretrained=False, num_classes=17*2).to('cuda')
# not using Hog loader

model = hg2(num_classes=17)
if torch.cuda.device_count() > 1:
  model = nn.DataParallel(model).to('cuda')

# model.to(device)


# model = hg2(num_classes=17).to('cuda')
train_data = hum36m_dataloader(root_data_path, annotation_path_train, True, [1.1, 2.0], False, 5, flip_prob = 1)
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batchSize_train,
                                                shuffle=True, num_workers=workers, pin_memory=False)
test_data = hum36m_dataloader(root_data_path, annotation_path_test, True, [1.1, 2.0], False, 5, flip_prob = 1)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batchSize_test,
                                                  shuffle=False, num_workers=workers, pin_memory=False)


total_param=0
all_param = []

all_param += list(model.parameters())
total_param += sum(p.numel() for p in model.parameters())
if optimizer_name == 'SGD':
    optimizer = optim.SGD(all_param, lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0001)
elif optimizer_name == 'Adam':
    optimizer = optim.Adam(all_param, lr=lr, amsgrad=True)
# optimizer_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
print("==> Total parameters: {:.2f}M".format(total_param / 1000000.0))

#loss function
criterion = {}
criterion['MSE'] = nn.MSELoss(size_average=True).cuda()
criterion['L1'] = nn.L1Loss(size_average=True).cuda()
criterion_MSE = nn.MSELoss()

#experiment tracker

if wandb_flag :
    import wandb
    wandb.login()
    wandb.init(project="new_seperate_modules", entity="nafisur")
    wandb.run.name = run_name
    wandb.run.save()
#load model 

if wandb_flag:
    wandb.watch(model)
#model size
print_model_size(model)

best_epoch = -1
for epoch in range(epoch_train):
    print('======>>>>> Online epoch: #%d <<<<<======' % (epoch + 1))
    torch.cuda.synchronize()
    if train_flag :
        model.train()
        timer = time.time()
        print('======>>>>> training <<<<<======')
        print('learning rate %f' % (lr))
        error_sum = AccumLoss()

        # mean_error = train(opt, actions, train_dataloader, model, criterion, optimizer)
        for i, data in enumerate(tqdm(train_dataloader, 0)):
            input_image = data['image']
            input_image = input_image.permute(0,3,1,2).to('cuda') #hwc
            N = input_image.size(0)            #[256, 2, 1, 17, 1] -> input image
            # initial_2d,features = model(input_image)  #[256, 3, 1, 17, 1] -> pred_out
            predicted_heatmap = model(input_image)
            # gt_3d = data['kp_3d'].to('cuda') #[256, 17, 3]
            # predicted_out3d = predicted_out3d.permute(0, 2, 3, 4, 1).contiguous().view(N, -1, 17, 3)
            # gt_3d = gt_3d.unsqueeze(1)
            gt_2d = data['kp_2d'].to('cuda')
            gt_heatmap = data['heatmap'].to('cuda')
            # calculate loss
            loss = joints_mse_loss(predicted_heatmap[-1],gt_heatmap)*10000 *1000
            # loss = criterion_MSE(predicted_heatmap, gt_heatmap) * 1000 # new loss factor
            # loss_initial_2d = 0
            # loss_2d = criterion_MSE(predicted_out2d, gt_2d)
            # loss_2d = 0
            # loss = 0.1 * loss_initial_2d + 0.1 * loss_2d + (1 - 0.01)*mpjpe(predicted_out3d, gt_3d) + 0.01*criterion['L1'](predicted_out3d, gt_3d)
            # print(f'prredicted')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # joint_error = criterion_MSE(initial_2d, gt_2d).item()
            # joint_error = criterion_MSE(predicted_heatmap, gt_heatmap).item()
            joint_error = joints_mse_loss(predicted_heatmap[-1], gt_heatmap).item()
            # error_sum.update(joint_error*N, N)
            error_sum.update(joint_error, N)
            # print(f'mean joint error(training): {error_sum.avg *1000}')
            # import pdb;pdb.set_trace()
        # e = error_sum.avg * 1000
        e = error_sum.avg *10000 *100
        print(f'mean joint error 2d(training): {e}')
        timer = time.time() - timer
        timer = timer / len(train_data)
        print('==> time to learn 1 sample = %f (ms)' % (timer * 1000))
        if wandb_flag:
            wandb.log({'taining error 2d': e, 'epoch':epoch})

    if test_flag :
        model.eval()
        error_sum = AccumLoss()
        timer = time.time()
        print('======>>>>> test <<<<<======')

        for i,data in enumerate(tqdm(test_dataloader, 0)):
            input_image = data['image']
            input_image = input_image.permute(0,3,1,2).to('cuda') #hwc
            N = input_image.size(0)
            predicted_heatmap = model(input_image)  #[256, 3, 1, 17, 1]
            # gt_3d = data['kp_3d'].to('cuda') #[256, 17, 3]
            gt_2d = data['kp_2d'].to('cuda')
            gt_heatmap = data['heatmap'].to('cuda')

            # predicted_out3d = predicted_out3d.permute(0, 2, 3, 4, 1).contiguous().view(N, -1, 17, 3) #[128, 17, 3]
            # print(f'gt_3d.shape {gt_3d.shape}')
            # gt_3d = gt_3d.unsqueeze(1)
            # joint_error = mpjpe(predicted_out3d, gt_3d).item()
            joint_error = joints_mse_loss(predicted_heatmap[-1], gt_heatmap).item()
            # joint_error = mpjpe(predicted_out3d, gt_3d)
            # error_sum.update(joint_error*N, N)
            error_sum.update(joint_error, N)
        final_error = error_sum.avg *10000 *100 # check if needed to be multiplied by 1000.0
        print(f'for test mean error 2d(testing) {final_error}')
        if wandb_flag:
            wandb.log({'testing error 2d': final_error})
        timer = time.time() - timer
        timer = timer / len(test_data)
        # if final_error < best_error:
        #     best_error = final_error
        #     best_epoch = epoch
        print('==> time to infer 1 sample = %f (ms)' % (timer * 1000))
        '''
        insert the save model code here from modulated GCN
        '''
        if final_error < best_error :
            best_error = final_error
            best_epoch = epoch + 1
        
            if save_model_flag :
                # previous_module_gcn_name = save_model(previous_module_gcn_name, save_dir, epoch, save_out_type, final_error, model, model_name)
                save_model_name = f'{model_name}_epoch_{epoch+1}.pth'
                path = os.path.join(save_dir, save_model_name)
                torch.save(model.state_dict(), path)
            # best_error = final_error
            # best_epoch = epoch + 1
        ######chech the decay code 
        if (epoch+1) % large_decay_epoch == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay
                lr *= lr_decay
        #schedular is used
        # scheduler.step(final_error)
    
print(f'best mpjpe on test data is {best_error} achieved at {best_epoch} epoch')