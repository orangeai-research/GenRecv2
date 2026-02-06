# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Run application
##########################
"""
from logging import getLogger
from itertools import product
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.logger import init_logger
from utils.configurator import Config
from utils.utils import init_seed, get_model, get_trainer, dict2str
import platform
import os

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed. Install with: pip install wandb")


def quick_start(model, dataset, config_dict, save_model=True, mg=False):
    # merge config dict
    config = Config(model, dataset, config_dict, mg)
    init_logger(config)
    logger = getLogger()
    # print config infor
    logger.info('██Server: \t' + platform.node())
    logger.info('██Dir: \t' + os.getcwd() + '\n')
    logger.info(config)

    # load data
    dataset = RecDataset(config)
    # print dataset statistics
    logger.info(str(dataset))

    train_dataset, valid_dataset, test_dataset = dataset.split()
    logger.info('\n====Training====\n' + str(train_dataset))
    logger.info('\n====Validation====\n' + str(valid_dataset))
    logger.info('\n====Testing====\n' + str(test_dataset))

    # item popularity analysis
    iid_field = config['ITEM_ID_FIELD']
    #print("iid_field:", iid_field) # itemID
    train_df = train_dataset.df
    # print("train_df:", train_df)
    item_counts = train_df[iid_field].value_counts() # 默认按计数从高到低降序排序（这是后续能直接取前 20% 的关键），索引为物品 ID，值为对应交互频次。
    # print("item_counts:", item_counts)
    '''
    train_df:         
            userID  itemID
    0            0    1587
    1            0    1879
    2            0       0
    5            1    2828
    6            1    5403
    ...        ...     ...
    160783   19443    7032
    160784   19443    7028
    160787   19444    7022
    160788   19444    6959
    160789   19444    7005

    item_counts: itemID
    938     554
    2459    472
    1431    438
    303     368
    2322    348
    unique_items: [938, 2459, 1431, 303, 2322...]
    '''
    unique_items = item_counts.index.tolist() # 提取所有唯一物品 ID，并转为列表格式。 即按交互频次降序排列的物品 ID
    # print("unique_items:", unique_items) 
    num_items = len(unique_items) # 唯一物品总数
    num_pop = int(num_items * 0.2) #  计算主流物品的数量（取唯一物品总数的 20%）。
    pop_items = set(unique_items[:num_pop]) # 筛选出前 20% 的主流物品ID，并转为集合（set）格式。
    config['pop_items'] = pop_items # Popular items ID: {1, 4, 9, 10, 14, 22...}
    # logger.info(f"Popular items: {config['pop_items']}")
    logger.info(f'Train dataset All Interaction items count: {num_items}, Popular items count: {len(pop_items)}, Niche items count: {num_items - len(pop_items)}')
    
    # user cold-start analysis
    uid_field = config['USER_ID_FIELD']
    # print("uid_field:", uid_field) # userID
    user_counts = train_df[uid_field].value_counts() # 每个用户交互的频率
    
    # Define thresholds
    zero_shot_threshold = 0
    few_shot_threshold = 3
    cold_start_threshold = 5
    
    # 1. Zero-shot users (in training set, interaction = 0? usually 0 but for safety check)
    # Actually in training set all users have at least 1 interaction usually.
    # But we define groups based on training history count.
    
    # 2. Few-shot users (1 <= count <= 3)
    few_shot_users = user_counts[(user_counts > 0) & (user_counts <= few_shot_threshold)].index.tolist()
    config['few_shot_users'] = set(few_shot_users)
    
    # 3. Standard Cold-start users (1 <= count <= 5)
    # Note: This overlaps with few-shot
    # cold_users = user_counts[user_counts <= cold_start_threshold].index.tolist()
    # config['cold_users'] = set(cold_users)
    
    # 4. Warm users (count > 5)
    warm_users = user_counts[user_counts > cold_start_threshold].index.tolist()
    config['warm_users'] = set(warm_users) # 筛选出交互次数>5的用户
    
    # Statistics
    n_warm = len(warm_users)
    n_few_shot = len(few_shot_users)
    n_total_train_users = len(user_counts)
    n_cold_in_train = n_total_train_users - n_warm

    #logger.info(f"warm_users: {config['warm_users']}")
    logger.info(f'User Grouping based on Training History:')
    logger.info(f'  Warm Users (> {cold_start_threshold} interactions): {n_warm}')
    logger.info(f'  Cold Users (<= {cold_start_threshold} interactions): {n_cold_in_train}')
    logger.info(f'  Few-shot Users (<= {few_shot_threshold} interactions): {n_few_shot}')


    # wrap into dataloader
    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    (valid_data, test_data) = (
        EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']),
        EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']))

    ############ Dataset loadded, run model
    hyper_ret = []
    val_metric = config['valid_metric'].lower()
    best_test_value = 0.0
    idx = best_test_idx = 0

    logger.info('\n\n=================================\n\n')

    # hyper-parameters
    hyper_ls = []
    if "seed" not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
    for i in config['hyper_parameters']:
        hyper_ls.append(config[i] or [None])
    # combinations
    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)
    for hyper_tuple in combinators:
        # random seed reset
        for j, k in zip(config['hyper_parameters'], hyper_tuple):
            config[j] = k
        init_seed(config['seed'], deterministic= True)

        logger.info('========={}/{}: Parameters:{}={}======='.format(
            idx+1, total_loops, config['hyper_parameters'], hyper_tuple))

        # Initialize wandb for this run
        use_wandb = config['use_wandb'] and WANDB_AVAILABLE
        if use_wandb:
            # Create run name
            run_name = f"{config['model']}_{config['dataset']}_seed{config['seed']}"
            if len(hyper_tuple) > 1:  # Has hyperparameters beyond seed
                param_str = "_".join([f"{k}{v}" for k, v in zip(config['hyper_parameters'][1:], hyper_tuple[1:])])
                run_name += f"_{param_str}"

            # Prepare config dict for wandb (exclude non-serializable objects)
            wandb_config = {}
            for k, v in config.final_config_dict.items():
                # Skip non-serializable objects like torch.device
                if k == 'device':
                    wandb_config[k] = str(v)
                elif not callable(v):
                    try:
                        # Try to convert to JSON-serializable type
                        wandb_config[k] = v
                    except:
                        wandb_config[k] = str(v)

            # Initialize wandb
            wandb_project = config['wandb_project'] if 'wandb_project' in config and config['wandb_project'] else 'GenMMRec'
            wandb.init(
                project=wandb_project,
                name=run_name,
                config=wandb_config,
                reinit=True,  # Allow multiple runs in same script
                tags=[config['model'], config['dataset']],
                notes=f"Training {config['model']} on {config['dataset']}"
            )
            logger.info(f"W&B run initialized: {run_name}")

        # set random state of dataloader
        train_data.pretrain_setup()
        # model loading and initialization
        model = get_model(config['model'])(config, train_data).to(config['device'])
        logger.info(model)

        # trainer loading and initialization
        # trainer = get_trainer()(config, model, mg)
        trainer = get_trainer(config['model'])(config, model, mg) 
        # debug
        # model training
        best_valid_score, best_valid_result, best_test_upon_valid = trainer.fit(train_data, valid_data=valid_data, test_data=test_data, saved=save_model)
        #########
        hyper_ret.append((hyper_tuple, best_valid_result, best_test_upon_valid))

        # Log final results to wandb
        if use_wandb:
            wandb.log({
                'best_valid_score': best_valid_score,
                **{f'valid/{k}': v for k, v in best_valid_result.items()},
                **{f'test/{k}': v for k, v in best_test_upon_valid.items()}
            })
            # Create summary
            wandb.run.summary.update({
                'best_valid_score': best_valid_score,
                **{f'final_valid_{k}': v for k, v in best_valid_result.items()},
                **{f'final_test_{k}': v for k, v in best_test_upon_valid.items()}
            })
            wandb.finish()

        # save best test
        if best_test_upon_valid[val_metric] > best_test_value:
            best_test_value = best_test_upon_valid[val_metric]
            best_test_idx = idx
        idx += 1

        logger.info('best valid result: {}'.format(dict2str(best_valid_result)))
        logger.info('test result: {}'.format(dict2str(best_test_upon_valid)))
        logger.info('████Current BEST████:\nParameters: {}={},\n'
                    'Valid: {},\nTest: {}\n\n\n'.format(config['hyper_parameters'],
            hyper_ret[best_test_idx][0], dict2str(hyper_ret[best_test_idx][1]), dict2str(hyper_ret[best_test_idx][2])))

    # log info
    logger.info('\n============All Over=====================')
    for (p, k, v) in hyper_ret:
        logger.info('Parameters: {}={},\n best valid: {},\n best test: {}'.format(config['hyper_parameters'],
                                                                                  p, dict2str(k), dict2str(v)))

    logger.info('\n\n█████████████ BEST ████████████████')
    logger.info('\tParameters: {}={},\nValid: {},\nTest: {}\n\n'.format(config['hyper_parameters'],
                                                                   hyper_ret[best_test_idx][0],
                                                                   dict2str(hyper_ret[best_test_idx][1]),
                                                                   dict2str(hyper_ret[best_test_idx][2])))

