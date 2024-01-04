import torch
from torch.utils.data import Dataset
import clip
from PIL import Image
from PIL import ImageFile
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, BatchSampler
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

LABELS = ["Provide your labels"]
device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
model, preprocess = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label:LABELS.index(label)
                                 for label in LABELS}
        self.label_to_indices2 = {label: np.where(np.asarray(self.labels) == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices2[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices2[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices2[class_]):
                    np.random.shuffle(self.label_to_indices2[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size

class TomAndJerryDataset(Dataset):
    def __init__(self, images, labels, captions):
        self.preprocess = preprocess
        self.images = images
        self.labels = labels
        self.captions = captions
        
    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        
        image = preprocess(Image.open(self.images[idx]))
        caption = clip.tokenize(self.captions[idx])
        label = LABELS.index(self.labels[idx])
        return image, caption, label

if __name__ == "__main__":
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.model_selection import StratifiedKFold

    print("Data Engineering")
    df = pd.read_csv("data.csv")
    df['folds'] = 0
    for idx, (train_idx, val_idx) in enumerate(StratifiedKFold(n_splits=5).split(df.paths.values, df.category.values)):
        df.loc[val_idx, "folds"] = idx
    
    train_df = df[df.folds != 0].reset_index(drop=True)
    test_df = df[df.folds == 0].reset_index(drop=True)

    train_image_paths = train_df.paths.values.tolist()
    train_captions = train_df.caption.values.tolist()
    train_labels = train_df.category.values.tolist()

    print("Creating Pytorch Dataset")
    train_dataset = TomAndJerryDataset(images=train_image_paths, labels=train_labels, captions=train_captions)
    # test_dataset = ProductDataset(xtest, ytest)

    print("Loading Custom made Sampler")
    train_sampler = BalancedBatchSampler(train_labels, 8, 1)
    train_loader = DataLoader(train_dataset, pin_memory=True, batch_sampler=train_sampler)
    # test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, pin_memory=True)

    print("Loading Loss Functions")
    train_loss_img = nn.CrossEntropyLoss()
    train_loss_txt = nn.CrossEntropyLoss()
    test_loss_img = nn.CrossEntropyLoss()
    test_loss_txt = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-3,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)


    EPOCHS = 5
    result_df = pd.DataFrame(index=np.arange(EPOCHS), columns=['train_epoch_loss', 'test_epoch_loss'])

    
    run_test_loss = 0.0


    print("Started Training....")
    for epoch in range(EPOCHS):
        
        run_train_loss = 0.0
        run_image_loss = 0.0
        run_text_loss = 0.0

        print(f"Epoch:{epoch}/5")
        model.train()
        for idx, batch in enumerate(train_loader):
            
            optimizer.zero_grad()

            images, captions, label = batch 

            images= images.to(device)
            captions = captions.to(device)
            captions = captions.squeeze(1)
            # print(images.shape, captions.shape)
            logits_per_image, logits_per_text = model(images, captions)

            ground_truth = torch.arange(len(images),dtype=torch.long,device=device)

            image_loss = train_loss_img(logits_per_image,ground_truth)
            text_loss = train_loss_txt(logits_per_text,ground_truth)
            total_loss = (image_loss + text_loss) / 2
            total_loss.backward()

            run_train_loss += total_loss.item()
            run_image_loss += image_loss.item()
            run_text_loss += text_loss.item()

            if idx % 100 == 0:
                print(f"Step_image_loss:{image_loss.item()}, Step_text_loss:{text_loss.item()}, Train Step Loss: {total_loss.item()}")

        print(f"Epoch_image_loss: {run_image_loss / len(train_loader)}, Epoch_text_loss: {run_test_loss/len(train_loader)},Train Epoch Loss: {run_train_loss / len(train_loader)}")
        # result_df.loc[epoch, "train_epoch_loss"] = run_train_loss / len(train_loader)
        
        # model.eval()
        # for idx, test_batch in test_loader:

        #     image, texts = batch
        #     images= images.to(device)
        #     texts = texts.to(device)

        #     logits_per_image, logits_per_text = model(images, texts)

        #     ground_truth = torch.arange(len(images),dtype=torch.long,device=device)

        #     total_test_loss = (test_loss_img(logits_per_image,ground_truth) + test_loss_txt(logits_per_text,ground_truth))/2
        #     run_test_loss += total_test_loss.item()
        # result_df.loc[epoch, "test_epoch_loss"] = run_test_loss / len(test_loader)
        
        # result_df.to_csv("results.csv", index=False)