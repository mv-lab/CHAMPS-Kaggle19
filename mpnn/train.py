from dataset import *
from model import *
from common import *
import os
import gc
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import faulthandler; faulthandler.enable()

def do_valid(net, valid_loader, loss_func=log_l1_loss):
    valid_num = 0
    valid_predict = []
    valid_coupling_type = []
    valid_coupling_value = []

    valid_loss = 0
    for b, (node, edge, edge_index, node_index, coupling_value, coupling_index, infor) in enumerate(valid_loader):
        net.eval()
        node = node.cuda()
        edge = edge.cuda()
        edge_index = edge_index.cuda()
        node_index = node_index.cuda()

        coupling_value = coupling_value.cuda()
        coupling_index = coupling_index.cuda()

        with torch.no_grad():
            predict = net(node, edge, edge_index, node_index, coupling_index)
            loss = loss_func(predict, coupling_value)

        # ---
        batch_size = len(infor)
        valid_predict.append(predict.data.cpu().numpy())
        valid_coupling_type.append(coupling_index[:, 2].data.cpu().numpy())
        valid_coupling_value.append(coupling_value.data.cpu().numpy())

        valid_loss += batch_size*loss.item()
        valid_num += batch_size

        print('\r %8d /%8d' %
              (valid_num, len(valid_loader.dataset)), end='', flush=True)

        pass  # -- end of one data loader --
    assert(valid_num == len(valid_loader.dataset))

    valid_loss = valid_loss/valid_num

    # compute
    predict = np.concatenate(valid_predict)
    coupling_value = np.concatenate(valid_coupling_value)
    coupling_type = np.concatenate(valid_coupling_type).astype(np.int32)
    mae, log_mae = compute_kaggle_metric(
        predict, coupling_value, coupling_type,)

    num_target = NUM_COUPLING_TYPE
    for t in range(NUM_COUPLING_TYPE):
        if mae[t] is None:
            mae[t] = 0
            log_mae[t] = 0
            num_target -= 1

    mae_mean, log_mae_mean = sum(mae)/num_target, sum(log_mae)/num_target

    valid_loss = log_mae + [valid_loss, mae_mean, log_mae_mean, ]
    return valid_loss


def run_train(lr=0.001, loss_func=log_l1_loss, num_iters=300000, batch_size=20, initial_checkpoint=None, 
              split_train='train_split_by_mol.80003.npy', split_valid='valid_split_by_mol.5000.npy', graph_dir='all_types',
              out_dir='data/results/zzz', coupling_types=['1JHC', '2JHC', '3JHC', '1JHN', '2JHN', '3JHN', '2JHH', '3JHH']):
    # setup  -----------------------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir + '/checkpoint', exist_ok=True)
    os.makedirs(out_dir + '/train', exist_ok=True)
    os.makedirs(out_dir + '/backup', exist_ok=True)
    backup_project_as_zip(PROJECT_PATH, out_dir + '/backup/code.train.%s.zip' % IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.train.txt', mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\t%s\n' % COMMON_STRING)
    log.write('\n')

    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\t__file__     = %s\n' % __file__)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')

    # dataset ----------------------------------------
    log.write('** dataset setting **\n')

    train_dataset = ChampsDataset(
        csv='train',
        mode='train',
        split=split_train,
        coupling_types=coupling_types,
        graph_dir=graph_dir
    )
    train_loader = DataLoader(
        train_dataset,
        #sampler=SequentialSampler(train_dataset),
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size,
        drop_last=True,
        num_workers=32,
        pin_memory=True,
        collate_fn=null_collate
    )

    valid_dataset = ChampsDataset(
        csv='train',
        mode='train',
        # split='debug_split_by_mol.1000.npy',
        split=split_valid,
        augment=None,
        coupling_types=coupling_types,
        graph_dir=graph_dir
    )
    valid_loader = DataLoader(
        valid_dataset,
        #sampler=SequentialSampler(valid_dataset),
        sampler=RandomSampler(valid_dataset),
        batch_size=batch_size,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=null_collate
    )

    assert(len(train_dataset) >= batch_size)
    log.write('batch_size = %d\n' % (batch_size))
    log.write('train_dataset : \n%s\n' % (train_dataset))
    log.write('valid_dataset : \n%s\n' % (valid_dataset))
    log.write('\n')

    # net ----------------------------------------
    log.write('** net setting **\n')

    #net = SagPoolLargerNet(node_dim=NODE_DIM, edge_dim=EDGE_DIM, num_target=NUM_TARGET).cuda()
    net = Set2SetLargerNet(node_dim=NODE_DIM, edge_dim=EDGE_DIM, num_target=NUM_TARGET).cuda()

    net.apply(weights_init)

    log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
    if initial_checkpoint is not None:
        checkpoint_iter = int(initial_checkpoint.split('/')[-1][:-10])
        net.load_state_dict(torch.load(initial_checkpoint,
                                       map_location=lambda storage, loc: storage))
    else:
        checkpoint_iter = -1

    log.write('%s\n' % (type(net)))
    log.write('\n')

    # optimizer ----------------------------------
    # freeze
    #     for p in net.encoder1.parameters(): p.requires_grad = False

    # net.set_mode('train',is_freeze_bn=True)
    # -----------------------------------------------

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()))
    #optimizer = torch.optim.RMSprop(net.parameters(), lr=0.0005, alpha=0.95)
    #optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=scheduler(0), momentum=0.9, weight_decay=0.0001)

    #scheduler = NullScheduler(lr=lr)

    #scheduler = StepScheduler([(checkpoint_iter,       0.000015),  
    #                           (checkpoint_iter+10000, 0.000012),  
    #                           (checkpoint_iter+20000, 0.00001),  
    #                           (checkpoint_iter+40000, 0.00008), 
    #                           (checkpoint_iter+60000, 0.00006), 
    #                           (checkpoint_iter+80000, 0.00005)])
    #print(scheduler.steps, checkpoint_iter)

    scheduler = OneCycleLR(optimizer, max_lr=lr, div_factor=25, pct_start=0.3, total_steps=num_iters)

    iter_accum = 1
    iter_smooth = 50
    iter_log = 500
    iter_valid = 500
    iter_save = [0, num_iters-1] + list(range(0, num_iters, 2500))  # 1000
    
    start_iter = 0
    start_epoch = 0
    if initial_checkpoint is not None:
        initial_optimizer = initial_checkpoint.replace(
            '_model.pth', '_optimizer.pth')
        if os.path.exists(initial_optimizer):
            checkpoint = torch.load(initial_optimizer)
            start_iter = checkpoint['iter']
            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            #adjust_betas(optimizer, 0.9, 0.999)
            del checkpoint
            gc.collect()

    log.write('optimizer\n  %s\n' % (optimizer))
    log.write('scheduler\n  %s\n' % (scheduler))
    log.write('\n')

    ## start training here! ##############################################

    log.write('** start training here! **\n')
    log.write('   batch_size =%d,  iter_accum=%d\n' % (batch_size, iter_accum))
    log.write('                       |--------------- VALID ----------------------------------------------------------------|-- TRAIN/BATCH ---------\n')
    log.write('                       |std %4.1f    %4.1f    %4.1f    %4.1f    %4.1f    %4.1f    %4.1f   %4.1f  |                    |        | \n' % tuple(COUPLING_TYPE_STD))
    log.write('lr        iter   epoch |    1JHC,   2JHC,   3JHC,   1JHN,   2JHN,   3JHN,   2JHH,   3JHH |  loss  mae log_mae | loss   | time          \n')
    log.write('--------------------------------------------------------------------------------------------------------------------------------------\n')

    train_loss = np.zeros(20, np.float32)
    valid_loss = np.zeros(20, np.float32)
    batch_loss = np.zeros(20, np.float32)
    iteration = 0
    i = 0
    start = timer()
    while iteration < num_iters:
        sum_train_loss = np.zeros(20, np.float32)
        count = 0
        optimizer.zero_grad()
        for node, edge, edge_index, node_index, coupling_value, coupling_index, infor in train_loader:
            batch_size = len(infor)
            iteration = i + start_iter
            epoch = (iteration - start_iter) * batch_size / \
                len(train_dataset) + start_epoch

            if (iteration % iter_valid == 0):
                valid_loss = do_valid(net, valid_loader, loss_func=loss_func)
                torch.cuda.empty_cache()

            # learning rate scheduler -------------
            if lr < 0:
                log.write('Negative learning rate!\n')
                break

            #adjust_learning_rate(optimizer, lr)
            lr = get_learning_rate(optimizer)
            #lr = scheduler(iteration)

            # one iteration update  -------------
            # net.set_mode('train',is_freeze_bn=True)

            if (iteration % iter_log == 0):
                print('\r', end='', flush=True)
                asterisk = '*' if iteration in iter_save and iteration != checkpoint_iter else ' '
                log.write('%0.6f  %5.1f%s %5.1f |  %+0.3f, %+0.3f, %+0.3f, %+0.3f, %+0.3f, %+0.3f, %+0.3f, %+0.3f | %+5.3f %5.2f %+0.2f | %+5.3f | %s' % (
                    lr, iteration/1000, asterisk, epoch,
                    *valid_loss[:11],
                    train_loss[0],
                    time_to_str((timer() - start), 'min'))
                )
                log.write('\n')

            if iteration in iter_save and iteration != checkpoint_iter:
                torch.save(net.state_dict(), out_dir +
                           '/checkpoint/%08d_model.pth' % (iteration))
                torch.save({
                    'optimizer': optimizer.state_dict(),
                    'iter': iteration,
                    'epoch': epoch,
                }, out_dir + '/checkpoint/%08d_optimizer.pth' % (iteration))

            net.train()
            node = node.cuda()
            edge = edge.cuda()
            edge_index = edge_index.cuda()
            node_index = node_index.cuda()
            coupling_value = coupling_value.cuda()
            coupling_index = coupling_index.cuda()

            optimizer.zero_grad()
            predict = net(node, edge, edge_index, node_index, coupling_index)
            loss = loss_func(predict, coupling_value)

            (loss/iter_accum).backward()
            if (iteration % iter_accum) == 0:
                optimizer.step()
                scheduler.step(iteration)

            # print statistics  ------------
            batch_loss[:1] = [loss.item()]
            sum_train_loss += batch_loss
            count += 1
            if iteration % iter_smooth == 0:
                train_loss = sum_train_loss/count
                sum_train_loss = np.zeros(20, np.float32)
                count = 0

            print('\r', end='', flush=True)
            asterisk = ' '
            print('%0.6f  %5.1f%s %5.1f |  %+0.3f, %+0.3f, %+0.3f, %+0.3f, %+0.3f, %+0.3f, %+0.3f, %+0.3f | %+5.3f %5.2f %+0.2f | %+5.3f | %s' % (
                lr, iteration/1000, asterisk, epoch,
                *valid_loss[:11],
                batch_loss[0],
                time_to_str((timer() - start), 'min')), end='', flush=True)
            i = i+1
    log.write('\n')


# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    output_directory = get_path() + 'data/results/all_types_new_features'
    checkpoint_path = get_path() + 'data/results/all_types_new_features/checkpoint/00007500_model.pth'

    #'1JHC', '2JHC', '3JHC', '1JHN', '2JHN', '3JHN', '2JHH', '3JHH'
    coupling_types = ['1JHC', '2JHC', '3JHC', '1JHN', '2JHN', '3JHN', '2JHH', '3JHH']

    run_train(lr=0.0018, loss_func=log_l1_loss, num_iters=400*1000, batch_size=16, coupling_types=coupling_types,
              split_train='train_split_by_mol.80003.npy', split_valid='valid_split_by_mol.5000.npy', 
              initial_checkpoint=None, graph_dir='all_types_selected_features', out_dir=output_directory)

    print('\nsuccess!')
