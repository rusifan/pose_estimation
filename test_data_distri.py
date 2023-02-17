import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
from tqdm import tqdm
from model.resnet import resnet50, resnet10, resnet18
from model.stackedhour import hg2
import numpy as np
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from dataloader.heatmap import hum36m_dataloader
# from dataloader.loader_hog import hum36m_dataloader
from utils.loss import joints_mse_loss, accuracy
from utils.utils1 import *
from utils.eval_cal import mpjpe


num_layers = 2
annotation_path_train = "annotation_body3d/fps25/h36m_train.npz"
annotation_path_test = "annotation_body3d/fps25/h36m_test.npz"
root_data_path = "/netscratch/nafis/human-pose/dataset_human36_nos7_f25"
batchSize_train = 34
batchSize_test = 16 #256 if using 168G mem 576756 780040
workers = 0
wandb_flag = False
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
best_epoch = -1
previous_module_gcn_name = ""
save_dir = "./results/stacked_dataparallel"
save_out_type = "xyz"
large_decay_epoch = 4
lr_decay = 0.90
save_model_flag = 1
######################################################
torch.manual_seed(0)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def prepare_train(rank, world_size, batch_size=32, pin_memory=False, num_workers=0):
    dataset = hum36m_dataloader(root_data_path, annotation_path_train, True, [1.1, 2.0], False, 5, flip_prob = 1)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=0, drop_last=False, shuffle=False, sampler=sampler)
    
    return dataloader,len(dataset)

def prepare_test(rank, world_size, batch_size=32, pin_memory=False, num_workers=0):
    dataset = hum36m_dataloader(root_data_path, annotation_path_test, True, [1.1, 2.0], False, 5, flip_prob = 1)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=0, drop_last=False, shuffle=False, sampler=sampler)
    
    return dataloader, len(dataset)

def cleanup():
    dist.destroy_process_group()


def main(rank):
    world_size = 3
    lr = 5e-3 #learning rate 0.005

    # setup the process groups
    total_param=0
    all_param = []
    setup(rank, world_size)
    # prepare the dataloader
    train_dataloader, len_train = prepare_train(rank, world_size,batch_size=batchSize_train)
    test_dataloader, len_test = prepare_test(rank, world_size, batch_size=batchSize_test)
    
    # instantiate the model(it's your own model) and move it to the right device
    model = hg2(num_classes=17).to(rank)
    criterion_MSE = nn.MSELoss()
    
    # wrap the model with DDP
    # device_ids tell DDP where is your model
    # output_device tells DDP where to output, in our case, it is rank
    # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    #################### The above is defined previously
    all_param += list(model.parameters())
    total_param += sum(p.numel() for p in model.parameters())
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(all_param, lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0001)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(all_param, lr=lr, amsgrad=True)
# optimizer_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    # loss_fn = Your_Loss()
    # best_epoch = -1
    for epoch in range(epoch_train):
        print('======>>>>> Online epoch: #%d <<<<<======' % (epoch + 1))
        if train_flag :
            train_dataloader.sampler.set_epoch(epoch) 
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
                predicted_heatmap = model(input_image)
                gt_2d = data['kp_2d'].to('cuda')
                # gt_heatmap = data['heatmap'].to('cuda')
                gt_heatmap = data['heatmap'].to(rank)

                # calculate loss

                # loss = joints_mse_loss(predicted_heatmap[-1],gt_heatmap)*10000 *1000
                loss = criterion_MSE(predicted_heatmap[-1],gt_heatmap)*10000 *1000
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                joint_error = joints_mse_loss(predicted_heatmap[-1], gt_heatmap).item()
                error_sum.update(joint_error, N)
                # print(f'mean joint error(training): {error_sum.avg *1000}')
            e = error_sum.avg *10000 *100
            print(f'mean joint error 2d(training): {e}')
            timer = time.time() - timer
            timer = timer / len_train
            print('==> time to learn 1 sample = %f (ms)' % (timer * 1000))
            if wandb_flag:
                wandb.log({'taining error 2d': e, 'epoch':epoch})
        if test_flag :
            test_dataloader.sampler.set_epoch(epoch) 
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
                joint_error = joints_mse_loss(predicted_heatmap[-1], gt_heatmap).item()
                error_sum.update(joint_error, N)
            final_error = error_sum.avg *10000 *100 # check if needed to be multiplied by 1000.0
            print(f'for test mean error 2d(testing) {final_error}')
            if wandb_flag:
                wandb.log({'testing error 2d': final_error})
            timer = time.time() - timer
            timer = timer / len_test
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
    cleanup()



if __name__ == "__main__":
    adj = np.load('/netscratch/nafis/human-pose/real_time_pose/model/adj_4.npy')
    adj = torch.from_numpy(adj).to('cuda')
    if wandb_flag :
        import wandb
        wandb.login()
        wandb.init(project="new_seperate_modules", entity="nafisur")
        wandb.run.name = run_name
        wandb.run.save()
    import torch.multiprocessing as mp
    # suppose we have 3 gpus
    world_size = 3    
    mp.spawn(
        main,
        # args=(world_size),
        nprocs=world_size
    )