from data import *
from dataset import *
from model import *
from common import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

COUPLING_TYPE_STATS = [
    # type   #mean, std, min, max
    '1JHC',  94.9761528641869,   18.27722399839607,   66.6008,   204.8800,
    '2JHC',  -0.2706244378832,    4.52360876732858,  -36.2186,    42.8192,
    '3JHC',   3.6884695895355,    3.07090647005439,  -18.5821,    76.0437,
    '1JHN',  47.4798844844683,   10.92204561670947,   24.3222,    80.4187,
    '2JHN',   3.1247536134185,    3.67345877025737,   -2.6209,    17.7436,
    '3JHN',   0.9907298624944,    1.31538940138001,   -3.1724,    10.9712,
    '2JHH', -10.2866051639817,    3.97960190019757,  -35.1761,    11.8542,
    '3JHH',   4.7710233597359,    3.70498129755812,   -3.0205,    17.4841,
]
NUM_COUPLING_TYPE = len(COUPLING_TYPE_STATS)//5

COUPLING_TYPE_MEAN = [COUPLING_TYPE_STATS[i*5+1]
                      for i in range(NUM_COUPLING_TYPE)]
COUPLING_TYPE_STD = [COUPLING_TYPE_STATS[i*5+2]
                     for i in range(NUM_COUPLING_TYPE)]

def run_submit(loss_func=log_l1_loss, graph_dir='all_types', out_dir='data/submission/all_types',
               initial_checkpoint='data/results/all_types/checkpoint/00022500_model.pth', validation=False):
    out_dir = get_path() + out_dir

    csv_file = out_dir + '/submit/submit-%s-larger.csv' % (initial_checkpoint.split('/')[-1][:-4])

    # setup  -----------------------------------------------------------------------------
    os.makedirs(out_dir + '/checkpoint', exist_ok=True)
    os.makedirs(out_dir + '/submit', exist_ok=True)
    os.makedirs(out_dir + '/backup', exist_ok=True)
    backup_project_as_zip(PROJECT_PATH, out_dir +
                          '/backup/code.submit.%s.zip' % IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.submit.txt', mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\t%s\n' % COMMON_STRING)
    log.write('\n')

    # dataset ----------------------------------------------------
    log.write('** dataset setting **\n')
    batch_size = 16  # *2 #280*2 #256*4 #128 #256 #512  #16 #32

    #-------------------------------------------------------------
    if validation:
        test_dataset = ChampsDataset(
            csv='train',
            mode='train',
            # split='debug_split_by_mol.1000.npy',
            split='valid_split_by_mol.5000.npy',
            augment=None,
            graph_dir=graph_dir
        )
        test_loader = DataLoader(
            test_dataset,
            sampler=SequentialSampler(test_dataset),
            batch_size=batch_size,
            drop_last=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=null_collate
        )
    else:
        test_dataset = ChampsDataset(
            mode='test',
            csv='test',
            split=None,
            augment=None,
            graph_dir=graph_dir
        )

        test_loader = DataLoader(
            test_dataset,
            sampler=SequentialSampler(test_dataset),
            #sampler=RandomSampler(train_dataset),
            batch_size=batch_size,
            drop_last=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=null_collate
        )

    log.write('batch_size = %d\n' % (batch_size))
    log.write('test_dataset : \n%s\n' % (test_dataset))
    print(len(test_dataset))
    log.write('\n')

    # net ----------------------------------------
    log.write('** net setting **\n')
    net = Set2SetLargerNet(node_dim=NODE_DIM, edge_dim=EDGE_DIM, num_target=NUM_TARGET).cuda()

    log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
    net.load_state_dict(torch.load(get_path() + initial_checkpoint,
                                   map_location=lambda storage, loc: storage))

    log.write('%s\n' % (type(net)))
    log.write('\n')

    ## start testing here! ##############################################
    test_num = 0
    test_predict = []
    test_coupling_type = []
    test_coupling_value = []
    test_id = []

    test_loss = 0

    start = timer()
    for b, (node, edge, edge_index, node_index, coupling_value, coupling_index, infor) in enumerate(test_loader):

        net.eval()
        with torch.no_grad():
            node = node.cuda()
            edge = edge.cuda()
            edge_index = edge_index.cuda()
            node_index = node_index.cuda()
            coupling_index = coupling_index.cuda()
            coupling_value = coupling_value.cuda()

            predict = net(node, edge, edge_index, node_index, coupling_index)
            loss = loss_func(predict, coupling_value)

        # ---
        batch_size = len(infor)
        test_id.extend(list(np.concatenate([infor[b][2] for b in range(batch_size)])))

        test_predict.append(predict.data.cpu().numpy())
        test_coupling_type.append(coupling_index[:, 2].data.cpu().numpy())
        test_coupling_value.append(coupling_value.data.cpu().numpy())

        test_loss += loss.item()*batch_size
        test_num += batch_size

        print('\r %8d/%8d     %0.2f  %s' % (
            test_num, len(test_dataset), test_num/len(test_dataset),
              time_to_str(timer()-start, 'min')), end='', flush=True)

    assert(test_num == len(test_dataset))
    print('\n')

    predict = np.concatenate(test_predict)
    if test_dataset.mode == 'test':
        df = pd.DataFrame(list(zip(test_id, predict)), columns=[
                          'id', 'scalar_coupling_constant'])
        df.to_csv(csv_file, index=False)

        log.write('id        = %d\n' % len(test_id))
        log.write('predict   = %d\n' % len(predict))
        log.write('csv_file  = %s\n' % csv_file)

    # ---------------------------------------------------------------
    # for debug
    if test_dataset.mode == 'train':
        test_loss = test_loss/test_num
        
        valid_df = pd.read_csv(get_data_path() + 'train.csv')
        valid_df = valid_df[valid_df.id.isin(test_id)]

        coupling_value = np.concatenate(test_coupling_value)
        coupling_type  = np.concatenate(test_coupling_type).astype(np.int32)
        #denormalization----------------------------------------------
        truth = []
        j = 0
        for i, row in valid_df.iterrows():
            if j % 10000 == 0: print(j)
            truth.append(row.scalar_coupling_constant)
            pred = predict[j]
            types_mean = COUPLING_TYPE_MEAN[COUPLING_TYPE_STATS.index(row.type)//5]
            types_std = COUPLING_TYPE_STD[COUPLING_TYPE_STATS.index(row.type)//5]
            predict[j] = pred * types_std + types_mean
            j+=1
            
        loss = np.abs(predict-truth)
        loss = loss.mean()
        loss = np.log(loss)
        #--------------------------------------------------------------

        mae, log_mae = compute_kaggle_metric(predict, truth, coupling_type,)
        for t in range(NUM_COUPLING_TYPE):
            log.write('\tcoupling_type = %s\n'%COUPLING_TYPE[t])
            log.write('\tmae     =  %f\n'%mae[t])
            log.write('\tlog_mae = %+f\n'%log_mae[t])
            log.write('\n')
        log.write('\n')


        log.write('-- final -------------\n')
        log.write('\ttest_loss = %+f\n'%test_loss)
        log.write('\tmae       =  %f\n'%np.mean(mae))
        log.write('\tlog_mae   = %+f\n'%np.mean(log_mae))
        log.write('\n')


# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))
    initial_checkpoint = 'data/results/all_types_new_features/checkpoint/00055000_model.pth'
    out_dir='data/submission/all_types_new_features'

    run_submit(loss_func=log_l1_loss, graph_dir='all_types_new_features', out_dir=out_dir,
               initial_checkpoint=initial_checkpoint, validation=True)
    
    print('\nsuccess!')
