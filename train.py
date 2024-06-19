'''Pre-training of MAE. i.e. SSL training before any finetuning or probing.
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
import datetime
from loguru import logger
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity


from models.ViT import VisionTransformer
from load_dataset import LoadLabelledDataset
from utils import load_checkpoint, save_checkpoint
from visualize_prediction import VisualizePrediction
import cred

def main(args):


    DATETIME_NOW = datetime.datetime.now().replace(second=0, microsecond=0) #datetime without seconds & miliseconds.

    #Read the config file from args.
    with open(args.config, 'r') as configfile:
        config = yaml.load(configfile, Loader=yaml.FullLoader)
        print("Configuration read successful...")

    
    with open(args.logging_config, 'r') as logging_configfile:
        logging_config = yaml.load(logging_configfile, Loader=yaml.FullLoader)
        print("Logging configuration file read successful...")
        print("Initializing logger") 
    
    
    ###Logger initialization
    if logging_config['disable_default_loggers']:
        logger.remove(0)

    logging_formatter = logging_config['formatters'][config['env']] #set the environment for the logger.

    
    Path(f"{logging_config['log_dir']}").mkdir(parents=True, exist_ok=True)
    #to output to a file
    logger.add(f"{logging_config['log_dir']}{DATETIME_NOW}-{logging_config['log_filename']}",
                    level=logging_formatter['level'],
                    format=logging_formatter['format'],
                    backtrace=logging_formatter['backtrace'],
                    diagnose=logging_formatter['diagnose'],
                    enqueue=logging_formatter['enqueue'])

    #to output to the console.
    logger.add(sys.stdout,
                level=logging_formatter['level'],
                format=logging_formatter['format'],
                backtrace=logging_formatter['backtrace'],
                colorize=True,
                diagnose=logging_formatter['diagnose'],
                enqueue=logging_formatter['enqueue'])
   

    #@@@@@@@@@@@@@@@@@@@@@@@@@ Extract the configurations from YAML file @@@@@@@@@@@@@@@@@@@@@@

    #Data configurations.
    BATCH_SIZE = config['data']['batch_size']
    IMAGE_SIZE = config['data']['image_size']
    IMAGE_DEPTH = config['data']['image_depth']
    DATASET_FOLDER = config['data']['dataset_folder']
    NUM_WORKERS = config['data']['num_workers']
    SHUFFLE = config['data']['shuffle']
    USE_RANDOM_HORIZONTAL_FLIP = config['data']['use_random_horizontal_flip']
    RANDOM_AFFINE_DEGREES = config['data']['random_affine']['degrees']
    RANDOM_AFFINE_TRANSLATE = config['data']['random_affine']['translate']
    RANDOM_AFFINE_SCALE = config['data']['random_affine']['scale']
    COLOR_JITTER_BRIGHTNESS = config['data']['color_jitter']['brightness']
    COLOR_JITTER_HUE = config['data']['color_jitter']['hue']


    #Model configurations.
    MODEL_SAVE_FOLDER = config['model']['model_save_folder']
    MODEL_NAME = config['model']['model_name']
    MODEL_SAVE_FREQ = config['model']['model_save_freq']
    N_SAVED_MODEL_TO_KEEP = config['model']['N_saved_model_to_keep']
    TRANSFORMER_BLOCKS_DEPTH = config['model']['encoder_transformer_blocks_depth']
    EMBEDDING_DIM = config['model']['encoder_embedding_dim']
    MLP_RATIO = config['model']['encoder_mlp_ratio']
    NUM_HEADS = config['model']['encoder_num_heads']
    ATTN_DROPOUT_PROB = config['model']['attn_dropout_prob']
    FEEDFORWARD_DROPOUT_PROB = config['model']['feedforward_dropout_prob']

    #Training configurations
    DEVICE = config['training']['device']
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and DEVICE=='gpu' else 'cpu')
    LOAD_CHECKPOINT = config['training']['load_checkpoint']
    LOAD_CHECKPOINT_EPOCH = config['training']['load_checkpoint_epoch']
    END_EPOCH = config['training']['end_epoch']
    START_EPOCH = config['training']['start_epoch']
    COSINE_UPPER_BOUND_LR = config['training']['cosine_upper_bound_lr']
    COSINE_LOWER_BOUND_LR = config['training']['cosine_lower_bound_lr']
    WARMUP_START_LR = config['training']['warmup_start_lr']
    WARMUP_STEPS = config['training']['warmup_steps']
    NUM_EPOCH_TO_RESTART_LR = config['training']['num_epoch_to_restart_lr']
    WEIGHT_DECAY = config['training']['weight_decay']
    USE_BFLOAT16 = config['training']['use_bfloat16']
    USE_NEPTUNE = config['training']['use_neptune']
    USE_TENSORBOARD = config['training']['use_tensorboard']
    USE_PROFILER = config['training']['use_profiler']
    
    
    if USE_NEPTUNE:
        import neptune

        NEPTUNE_RUN = neptune.init_run(
                                        project=cred.NEPTUNE_PROJECT,
                                        api_token=cred.NEPTUNE_API_TOKEN
                                      )
        #we have partially unsupported types. Hence the utils method.
        NEPTUNE_RUN['parameters'] = neptune.utils.stringify_unsupported(config)


    if USE_TENSORBOARD:
        TB_WRITER = writer = SummaryWriter(f'runs/vit-{DATETIME_NOW}')

    logger.info("Init ViT model...")
    
    VIT_MODEL = VIT(patch_size=PATCH_SIZE, 
                  image_size=IMAGE_SIZE, 
                  image_depth=IMAGE_DEPTH,
                  encoder_embedding_dim=ENCODER_EMBEDDING_DIM, 
                  decoder_embedding_dim=DECODER_EMBEDDING_DIM, 
                  encoder_transformer_blocks_depth=ENCODER_TRANSFORMER_BLOCKS_DEPTH, 
                  decoder_transformer_blocks_depth=DECODER_TRANSFORMER_BLOCKS_DEPTH, 
                  masking_ratio=MASKING_RATIO,
                  normalize_pixel=NORMALIZE_PIXEL,
                  device=DEVICE,
                  encoder_mlp_ratio=ENCODER_MLP_RATIO, 
                  decoder_mlp_ratio=DECODER_MLP_RATIO,
                  encoder_num_heads=ENCODER_NUM_HEADS,
                  decoder_num_heads=DECODER_NUM_HEADS, 
                  attn_dropout_prob=ATTN_DROPOUT_PROB,
                  feedforward_dropout_prob=FEEDFORWARD_DROPOUT_PROB,
                  logger=logger).to(DEVICE, non_blocking=True)
    

    if USE_PROFILER:
        sample_inp = torch.randn(2, IMAGE_DEPTH, IMAGE_SIZE, IMAGE_SIZE, requires_grad=False)

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("Model inference"):
                VIT_MODEL(sample_inp)

        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))



    
    if USE_TENSORBOARD:
        sample_inp = torch.zeros(2, IMAGE_DEPTH, IMAGE_SIZE, IMAGE_SIZE, requires_grad=False).to(DEVICE)
        
        with torch.no_grad():
            TB_WRITER.add_graph(VIT_MODEL, sample_inp)



    DATASET_MODULE = LoadLabelledDataset(dataset_folder_path=DATASET_FOLDER, 
                                       image_size=224, 
                                       image_depth=3, 
                                       transforms=transforms.Compose([transforms.Resize((self.image_size, self.image_size)),
                                                                      transforms.ColorJitter(brightness=COLOR_JITTER_BRIGHTNESS, hue=COLOR_JITTER_HUE),
                                                                      transforms.RandomAffine(degrees=RANDOM_AFFINE_DEGREES, translate=RANDOM_AFFINE_TRANSLATE, scale=RANDOM_AFFINE_SCALE),
                                                                      transforms.ToTensor(),
                                                                      transforms.Lambda(lambda x: x.repeat(int(3/x.shape[0]), 1, 1)), #to turn grayscale arrays into compatible RGB arrays.
                                                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
                                       logger=logger)

    DATALOADER = DataLoader(DATASET_MODULE, 
                            batch_size=BATCH_SIZE, 
                            shuffle=SHUFFLE, 
                            num_workers=NUM_WORKERS)

    

    OPTIMIZER = torch.optim.AdamW(params=VIT_MODEL.parameters(), 
                                  lr=COSINE_UPPER_BOUND_WD, 
                                  weight_decay=WEIGHT_DECAY)

    SCHEDULER = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=OPTIMIZER, 
                                                                     T_0=NUM_EPOCH_TO_RESTART_LR, 
                                                                     T_mult=1, 
                                                                     eta_min=COSINE_LOWER_BOUND_LR)

    CRITERION = torch.nn.CrossEntropyLoss().to(DEVICE)

    SCALER = None

    if USE_BFLOAT16:
        SCALER = torch.cuda.amp.GradScaler()

    if LOAD_CHECKPOINT:
        VIT_MODEL, START_EPOCH = load_checkpoint(model_save_folder=MODEL_SAVE_FOLDER, 
                                                                    model_name=MODEL_NAME, 
                                                                    VIT_MODEL=VIT_MODEL,
                                                                    load_checkpoint_epoch=None, 
                                                                    logger=logger)
        for _ in range(START_EPOCH):
            SCHEDULER.step() #this is needed to restore the parameters of the optimizer. 




    
    for epoch_idx in range(START_EPOCH, END_EPOCH):

        logger.info(f"Training has started for epoch {epoch_idx}")
        
        VIT_MODEL.train() #set to train mode.

        epoch_loss = 0
        

        try:
            for idx, data in tqdm(enumerate(DATALOADER)):   
         

                images = data['images'].to(DEVICE)
                labels = data['labels'].to(DEVICE)

                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=USE_BFLOAT16):
                    preds = VIT_MODEL(x=images)
                    
                
                loss = CRITERION(Input=preds, Target=labels)

                #backward and step
                if USE_BFLOAT16:
                    SCALER.scale(loss).backward()
                    SCALER.step(OPTIMIZER)
                    SCALER.update()
                else:
                    loss.backward()
                    OPTIMIZER.step()
                
                OPTIMIZER.zero_grad()
                
                
                epoch_loss += loss.item()

            
            SCEHDULER.step()

            if USE_NEPTUNE:
                NEPTUNE_RUN['train/learning_rate'].append(OPTIMIZER.param_groups[0]["lr"])

            if USE_TENSORBOARD:
                TB_WRITER.add_scalar("LearningRate", OPTIMIZER.param_groups[0]["lr"], epoch_idx)


            
        except Exception as err:

            logger.error(f"Training stopped at epoch {epoch_idx} due to {err}")

            if USE_NEPTUNE: 
                NEPTUNE_RUN.stop() 

            if USE_TENSORBOARD:
                TB_WRITER.close()

            sys.exit()


        logger.info(f"The training loss at epoch {epoch_idx} is : {epoch_loss}")

        if USE_TENSORBOARD:
            TB_WRITER.add_scalar("Loss/train", epoch_loss, epoch_idx)
        
        if USE_NEPTUNE:
            NEPTUNE_RUN['train/loss_per_epoch'].append(epoch_loss)
            
        

        
        if epoch_idx % MODEL_SAVE_FREQ == 0:
            
            save_checkpoint(model_save_folder=MODEL_SAVE_FOLDER, 
                    model_name=MODEL_NAME, 
                    VIT_MODEL=VIT_MODEL, 
                    scaler=SCALER, 
                    epoch=epoch_idx, 
                    loss=epoch_loss, 
                    N_models_to_keep=N_SAVED_MODEL_TO_KEEP, 
                    logger=logger
                    )
        
    if USE_NEPTUNE:
        NEPTUNE_RUN.stop() 

    if USE_TENSORBOARD:
        TB_WRITER.close()
                                     
                                  

if __name__ == '__main__':

    # os.environ["OMP_NUM_THREADS"] = "1"
    # os.environ["MKL_NUM_THREADS"] = "1"

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str, help='Specify the YAML config file to be used.')
    parser.add_argument('--logging_config', required=True, type=str, help='Specify the YAML config file to be used for the logging module.')
    args = parser.parse_args()
    main(args)
