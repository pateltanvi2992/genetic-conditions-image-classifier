import argparse
from loguru import logger
import pathlib
import random
import numpy as np
import pandas as pd
import torch
import torchmetrics
import pandas as pd
import datasets
from sklearn.metrics import accuracy_score, f1_score
print("test")
def main(args):
    print('test in main')
    # reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms = True

    # set GPU device ID
    device = torch.device("cuda:%d"%args.device if torch.cuda.is_available() and args.device>=0 else "cpu")
    print(device)
    # Pre-calculated channel means (nearly same but still use them)
    mean_bgr = {'fold-1':[112.482, 123.050, 147.127],
                'fold-2':[112.475, 123.011, 147.073], 
                'fold-3':[112.359, 122.850, 147.066], 
                'fold-4':[112.912, 123.480, 147.665], 
                'fold-5':[112.554, 123.063, 147.243]}

    # Performance metrics
    df = pd.DataFrame(columns=['Accuracy','F1-Score (macro)', 'F1-Score (weighted)'])
    results = []
    table = []
    filenames, labels, predictions, probabilities, groups = [],[],[],[],[]
    print(df)
    for fold_idx, fold in enumerate(['fold-1','fold-2','fold-3','fold-4','fold-5']):

        # datasets
        # root dataset directory containing:
        # * ./images : containing raw images (N=3547)
        # * ./features/face-parser (segmentation maps of all images)
        # Segmentation maps are not for evalauting model performance, but later for XAI evaluation.
        test_dataset = datasets.NIHFacesDataset(root_dir=args.dataset_folder,
                                                metadata_file='./metadata/partitions_july16th.csv',
                                                fold=fold,
                                                split='val',
                                                mean_bgr=mean_bgr[fold],
                                                image_size=224,
                                                flip=False)

        # Load the models
        #from models.ResNets import ResNet50
        #model = ResNet50(model_path=None)
        print('for loop')
        # Read trained weights (not face recognition models, the ones trained on NIH Faces)
        #model_path = list(pathlib.Path('./results/%s/%s'%('VGGFace2_ResNet50', fold)).glob('epoch-01-*.pt'))[-1]
        val_acc = np.array([float(s.as_posix().split('-test_accuracy-')[-1].replace('.pt','')) for s in list(pathlib.Path('./results/%s/%s'%('VGGFace2_ResNet50', fold)).glob('epoch-*.pt'))])
        #val_acc = np.array([float(s.as_posix().split('-test_accuracy-')[-1].replace('.pt','')) for s in list(pathlib.Path('./results/%s/%s'%('VGGFace2_ResNet50', fold)).glob('epoch-*.pt'))])
        model_path = list(pathlib.Path('./results/%s/%s'%('VGGFace2_ResNet50', fold)).glob('epoch-*.pt'))[np.argmax(val_acc)].as_posix()
        print(model_path)
        model = torch.load(model_path)
        model.to(device)
        eval_mode = model.eval()
        logger.info('model_path: %s'%model_path)

        num_samples = len(test_dataset)
        print('Evaluating VGGFace2-pretrained ResNet50 on NIH Faces (%s, N=%d)'%(fold, num_samples))
        
        for index in range(0, num_samples):

            # load images and labels
            image, label, filename, _ = test_dataset.__getitem__(index)
            image = image.unsqueeze(0).to(device)

            # forward pass
            output = model(image)

            # calculate accuracy
            probs = torch.nn.functional.softmax(output, dim=1).detach().cpu()
            #print(probs)
            _, pred = torch.max(probs, dim=1)
            #print(filename)
            #print(_)
            #print(label)
            #print(pred)

            filenames.append(filename)
            labels.append(int(label))
            groups.append(fold)
            predictions.append(pred.numpy()[0])
            probabilities.append(probs.numpy()[0])
            table.append(_.numpy()[0])
            
            #image = image.rename(columns={0 : "filenames"})

        results.append({'method':'VGGFace2_ResNet50', 'labels': labels, 'predictions':predictions, 'probabilities':probabilities, 'groups':groups})
        
        df.loc[fold_idx] = [round(accuracy_score(labels, predictions), 3)] + \
                           [round(f1_score(labels, predictions, average='macro'), 3)] + \
                           [round(f1_score(labels, predictions, average='weighted'), 3)]

    print(repr(df))
    filenames = pd.DataFrame(filenames)
    #print(filenames)
    labels = pd.DataFrame(labels)
    predictions = pd.DataFrame(predictions)
    filenames = pd.DataFrame(filenames)
    labels = pd.DataFrame(labels)
    probabilities = pd.DataFrame(probabilities)
    table= pd.DataFrame(table)
    #print(probabilities)
    result_pred = pd.concat([filenames, labels, predictions, table, probabilities],axis=1)
    #print(result_pred)
    result_pred.to_csv("Omer_crop_val.csv")
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture',   default='VGGFace2_ResNet50',   type=str, help='experiment name')
    parser.add_argument('--seed',           default=42,      type=int, help='random seed')
    parser.add_argument('--device',         default=0,       type=int, help='device: cpu (-1), cuda: 0, 1')
    parser.add_argument('--project_root',   default='./',    type=str, help='project root')
    parser.add_argument('--dataset_folder', type=str, help='Root data directory containing images subfolder with all NIH-Faces.')
    parser.add_argument('--num_classes',    default=12,      type=int,    help='number of classes')

    parser.add_argument('--fold',           default='fold-1',type=str,    help='fold: 1-5')
    args = parser.parse_args()
    main(args)
