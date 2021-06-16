import argparse
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional.classification import auroc
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from sklearn.metrics import accuracy_score, multilabel_confusion_matrix, f1_score, precision_score, recall_score, precision_recall_fscore_support 
from itertools import compress

from pathlib import Path


# For group analysis:
# This assumes that there are sub-groups within the gold data that are found in the folder ./groups/
# Each file represents a sub-group and has and id per line that corresponds to the sample_id in the gold/test data
group_path = './groups/'
group_files = [n.name for n in Path(group_path).glob('*.txt')]

mapped_groups = dict()
groups = dict()

for group in group_files:
    groups[group] = [int(l) for l in open(group_path+group, 'r').readlines() if l.strip() != '']
    for i in groups[group]:
        if i not in mapped_groups:
            mapped_groups.update({i: list()})
        mapped_groups[i].append(group)

#For the different runs, use a random seed
RANDOM_SEED = np.random.choice(range(100))
#RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

errors = list()

BERT_MODEL_NAME = 'bert-base-cased'
tokeniser = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

#sample_row = df.iloc[16]
#sample_text = sample_row.text
#sample_labels = sample_row[LABEL_COLUMNS]

#    encoding = tokeniser.encode_plus(
#        sample,
#        truncation=True,
#        add_special_tokens=True,
#        max_length=128,
#        return_token_type_ids=False,
#        padding="max_length",
#        return_attention_mask=True,
#        return_tensors="pt"
#    )

#print(encoding.keys())
#print(encoding['input_ids'].shape)
#print(encoding['attention_mask'].shape)

#print(encoding['input_ids'].squeeze())
#print(encoding['attention_mask'].squeeze())

#print(tokeniser.convert_ids_to_tokens(encoding['input_ids'].squeeze())[:20])


def classify(sample, model, tokeniser, label_names, thresholds):

    encoding = tokeniser.encode_plus(
        sample,
        truncation=True,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt"
    )

    _, prediction = model(encoding['input_ids'], encoding['attention_mask'])
    prediction = prediction.flatten().numpy()
    # print(prediction)

    predicted_labels = []
    binarised_labels = []
    scores = []

    for i, label_name in enumerate(label_names):
        label_probability = prediction[i]
        scores.append(label_probability)
        if label_probability > thresholds[i]:
            # print('LABEL PROB', label_probability)
            predicted_labels.append(label_name)
            binarised_labels.append(1)
        else:
            binarised_labels.append(0)
            
    return predicted_labels, binarised_labels, scores


#train_dataset = TextDataset(train_df, tokeniser)
#sample_item = train_dataset[0]
#print('SAMPLE ITEM KEYS', sample_item.keys())
#print(sample_item['text'])
#print(sample_item['input_ids'].shape, sample_item['attention_mask'].shape, sample_item['labels'].shape)
#sample_item['labels']

# bert_model = BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)
# prediction = bert_model(sample_item['input_ids'].unsqueeze(dim=0), sample_item['attention_mask'].unsqueeze(dim=0))

parser = argparse.ArgumentParser(description='args for the multilabel text classifier')
parser.add_argument('-dropout', type=float, default=0.1)
parser.add_argument('-early', action='store_true')
parser.add_argument('-folds', type=int, default=10)
parser.add_argument('-epochs', type=int, default=20)
parser.add_argument('-batch_size', type=int, default=32)
parser.add_argument('-csv', type=str, default='test.csv')
parser.add_argument('-var_thresh', action='store_true', default=False)
parser.add_argument('-threshold', type=float, default=0.5)
parser.add_argument('-soft', action='store_true', default=False)
parser.add_argument('-ratings', action='store_true', default=False)
parser.add_argument('-test', type=str)
parser.add_argument('-theme', action='store_true', default=False)
parser.add_argument('-gold_val', action='store_true', default=False)
parser.add_argument('-output', type=str, default='./output/')
parser.add_argument('-cycles', type=int)
parser.add_argument('-print_sents', action='store_true')
parser.add_argument('-print_output', action='store_true')
parser.add_argument('-final_model', action='store_true')
parser.add_argument('-errors', action='store_true')

args = parser.parse_args()

OUTPUT = args.output

NUM_EPOCHS = args.epochs  # 10
BATCH_SIZE = args.batch_size
THRESHOLD = args.threshold
FOLDS = args.folds

df = pd.read_csv(args.csv)
print('### NUM ENTRIES AFTER READING CSV', len(df))

if args.test:
    df_test = pd.read_csv(args.test)
else: 
    df_test = pd.read_csv('test.csv')
df = df.sample(frac=1).reset_index(drop=True)
print('### NUM ENTRIES AFTER SHUFFLING', len(df))

splits = np.array_split(df, args.folds)
segments = deepcopy(splits)

if args.final_model:
    splits = df
    FOLDS = 'final'


LABEL_COLUMNS = list(df.keys())[2:]

print('LABEL_COLUMNS', LABEL_COLUMNS)
print('===\n')

l_thresholds = dict((l, [args.threshold]) for l in LABEL_COLUMNS)
thresholds = [THRESHOLD] * len(LABEL_COLUMNS)


class TextDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokeniser: BertTokenizer, max_token_len: int=128):
        self.data = data
        self.tokeniser = tokeniser
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):

        data_row = self.data.iloc[index]
        text = data_row.text
        labels = [data_row[l] for l in LABEL_COLUMNS]  # data_row[LABEL_COLUMNS]

        encoding = self.tokeniser.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        return dict(
            text=text,
            input_ids=encoding['input_ids'].flatten(),
            attention_mask=encoding['attention_mask'].flatten(),
            labels=torch.FloatTensor(labels)
            # labels=torch.LongTensor()
        )


class TextDataModule(pl.LightningDataModule):
    def __init__(self, train_df, test_df, tokeniser, batch_size=8, max_token_len=128):
        super().__init__()
        self.train_df = train_df
        self.test_df = test_df
        self.tokeniser = tokeniser
        self.batch_size = batch_size
        self.max_token_len = max_token_len

    def setup(self):
        self.train_dataset = TextDataset(
            self.train_df,
            self.tokeniser,
            self.max_token_len
        )

        self.test_dataset = TextDataset(
            self.test_df,
            self.tokeniser,
            self.max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=4)


class TextClassification(pl.LightningModule):
    def __init__(self, n_classes: int, steps_per_epoch=None, n_epochs=None, hidden_dropout_prob=0.1):
        super().__init__()

        # self.config = BertConfig.from_json_file('./init_config.json')
        self.config = BertConfig(vocab_size=28996, hidden_dropout_prob=hidden_dropout_prob, n_classes=n_classes, return_dict=True)
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME, config=self.config)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.steps_per_epoch = steps_per_epoch
        self.n_epochs = n_epochs
        self.predict = False
        # self.criterion = nn.CrossEntropyLoss()  
        self.criterion = nn.BCELoss()

        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask, labels=LABEL_COLUMNS):
        output = self.bert(input_ids, attention_mask)
        output = self.classifier(output.pooler_output)
        output = torch.sigmoid(output)
        loss = 0
        if labels is not None and self.predict == False:
            loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("training_loss", loss, logger=True)
        return {'loss': loss, 'predictions': outputs, 'labels': labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, logger=True)
        return loss

    def training_epoch_end(self, outputs):

        labels = []
        predictions = []

        for output in outputs:
            for out_labels in output['labels'].detach().cpu():
                labels.append(out_labels)

            for out_predictions in output['predictions'].detach().cpu():
                predictions.append(out_predictions)

        labels = torch.stack(labels)
        predictions = torch.stack(predictions)

#        for i, name in enumerate(LABEL_COLUMNS):
#            roc_score = auroc(predictions[:, i], labels[:, i])
#            self.logger.experiment.add_scalar(f'{name}_roc_aur/Train', roc_score, self.current_epoch)

    def configure_optimizers(self):

        optimizer = AdamW(self.parameters(), lr=2e-5)

        warmup_steps = self.steps_per_epoch // 3
        total_steps = self.steps_per_epoch * self.n_epochs - warmup_steps

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            warmup_steps,
            total_steps
        )

        return [optimizer], [scheduler]


if args.early:
    early = 'do_early'
else:
    early = 'no_early'

suffix = '/'
if args.soft:
    suffix = 'soft'+suffix
if args.theme:
    suffix = 'theme'+suffix
if args.ratings:
    suffix = 'ratings'+suffix
if args.gold_val:
    suffix = 'gold_val'+suffix
if args.var_thresh:
    suffix = 'var_thresh'+suffix
    
checkpoint_folder = '-'.join([str(NUM_EPOCHS), str(BATCH_SIZE), str(args.dropout), early, str(FOLDS), 'thresh'+str(THRESHOLD)]) + suffix

def update_thresholds(scores, true, ts):
    my0 = zip(LABEL_COLUMNS, ts)
    my = dict((l, [t]) for l, t in my0)
    my_thresholds = list()
    for j, s in enumerate(scores):
        for i, l in enumerate(LABEL_COLUMNS):
            if true[j][i] == 1:
                my[l].append(s[i])
    for l in LABEL_COLUMNS:
        new_mean = round(np.mean(my[l]), 2)
        my_thresholds.append(new_mean)
        l_thresholds[l].append(new_mean)
    return my_thresholds

def create_data_and_train(train_df, val_df, test_df, thresholds):
    data_module = TextDataModule(train_df, val_df, tokeniser, batch_size=BATCH_SIZE)
    data_module.setup()

    model = TextClassification(n_classes=len(LABEL_COLUMNS),
                                    steps_per_epoch=len(train_df) // BATCH_SIZE,
                                    n_epochs=NUM_EPOCHS,
                                    hidden_dropout_prob=args.dropout
                                   )

    early_stop_callback = EarlyStopping(
        # divergence_threshold=0.1,
        # check_val_every_n_epoch=2,
        monitor='val_loss',
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode='min'
    )
    all_pred = list()
    all_true = list()
    all_sents = list()
    all_labels = list()
    all_scores = list()
    all_ids = list()

    if args.early:
        trainer = pl.Trainer(callbacks=[early_stop_callback], max_epochs=NUM_EPOCHS, gpus=1)  # , progress_bar_refresh_rate=30)
    else:
        trainer = pl.Trainer(max_epochs=NUM_EPOCHS, gpus=1)

    trainer.fit(model, data_module)
    checkpoint_name = '-'.join([str(NUM_EPOCHS), str(BATCH_SIZE), str(args.dropout), early, str(FOLDS), 'thresh'+str(THRESHOLD)]) + '.ckpt' 
    #trainer.save_checkpoint("final_checkpoint.ckpt")
    print('OUTPUT', OUTPUT, checkpoint_folder, checkpoint_name)
    trainer.save_checkpoint(OUTPUT + checkpoint_folder + checkpoint_name)
    pickle.dump(LABEL_COLUMNS, open(OUTPUT + checkpoint_folder + 'labels.pkl', 'wb'))

    trained_model = TextClassification(n_classes=len(LABEL_COLUMNS)).load_from_checkpoint(OUTPUT + checkpoint_folder + checkpoint_name, n_classes=len(LABEL_COLUMNS))
    trained_model.freeze()
    trained_model.predict = True

    # steps_per_epoch=len(train_df)//BATCH_SIZE
    # test_text = "I am being evicted"

    for i in range(len(test_df)):
        test_text = test_df.iloc[i].text
        test_labels = [test_df.iloc[i][n] for n in LABEL_COLUMNS]
        test_id = test_df.iloc[i].sample_id
        # print('TYPE test_id', type(test_id))
        try:
            label_predictions, binarised, probs = classify(test_text, trained_model, tokeniser, LABEL_COLUMNS, thresholds)
            if args.print_output:
                print('## TEST OUTPUT')
                print(label_predictions)
                print(binarised)
                print(probs)
            all_labels.append(label_predictions)
            all_true.append(test_labels)
            all_pred.append(binarised)
            all_sents.append(test_text)
            all_scores.append(probs)
            all_ids.append(test_id)
        except Exception as e:
            errors.append((test_df.iloc[i].sample_id, str(e)))
    return all_labels, all_true, all_pred, all_sents, all_scores, all_ids

final_labels = list()
final_true = list()
final_pred = list()
final_sents = list()
final_scores = list()
final_ids = list()

cycles = len(splits)
if args.cycles and args.cycles < len(splits):
    cycles = args.cycles

for i in range(0, cycles):
    print('## FOLD', i, len(splits), i, FOLDS)

    temp_test = segments.pop(i)
    s_test_df = temp_test
    if args.test:
        my_tests = temp_test['sample_id'].tolist()
        s_test_df = df_test[df_test['sample_id'].isin(my_tests)]
    rest_df = pd.concat(segments)

    if args.final_model:
        rest_df = pd.concat(rest_df, s_test_df)

    s_train_df, s_val_df = train_test_split(rest_df, test_size=0.10)

    if args.final_model:
        s_test_df = s_val_df

    if args.gold_val:
        my_val = s_val_df['sample_id'].tolist()
        s_val_df = df_test[df_test['sample_id'].isin(my_val)]
    print(s_test_df.iloc[i].sample_id)
    print(len(s_test_df), len(s_train_df), len(s_val_df))
    all_ls, all_ts, all_ps, all_s0s, all_s1s, all_ids = create_data_and_train(s_train_df, s_val_df, s_test_df, thresholds)
    segments = deepcopy(splits)
    final_labels.extend(all_ls)
    final_true.extend(all_ts)
    final_pred.extend(all_ps)
    final_sents.extend(all_s0s)
    final_scores.extend(all_s1s)
    final_ids.extend(all_ids)
    my_true = np.array(all_ts) # .astype(np.float)
    my_pred = np.array(all_ps) # .astype(np.float)

    acc_score_i = accuracy_score(my_true, my_pred)
    print('ACCURACY SCORE FOLD', i, acc_score_i)
    if args.var_thresh:
        thresholds = update_thresholds(all_s1s, all_ts, thresholds)  # all_scores, all_ts
        pickle.dump(thresholds, open(OUTPUT + checkpoint_folder + 'thresholds.pkl', 'wb'))


if args.print_sents:
    for i in range(len(final_sents)):
        print('TEXT', final_sents[i])
        compressed_true = compress(LABEL_COLUMNS, final_true[i])
        compressed_scores = compress(final_scores[i], final_true[i])
        print('TRUE', list(compressed_true))
        print('PREDICTED', final_labels[i])
        print('SCORES', list(compressed_scores))
        print('---\n')

group_results = dict()    

for group in groups:
    indices = [i for i, x in enumerate(final_ids) if x in groups[group]]
    print('GROUP', group)
    print(groups[group])
    print('INDICES', indices)
    print([final_pred[p] for p in indices])
    group_results.update({group: 
                          {'true': np.array([final_true[p] for p in indices]),
                           'pred': np.array([final_pred[p] for p in indices])}})

group_results.update({'all.txt':
                      {'true': np.array(final_true),
                       'pred': np.array(final_pred)}})

    #y_true = np.array(final_true) # .astype(np.float)
    #y_pred = np.array(final_pred) # .astype(np.float)

for group in group_results:
    f = OUTPUT+checkpoint_folder+group
    writer = open(f, 'w')
    y_true = group_results[group]['true']
    y_pred = group_results[group]['pred']

    print(group)
    print(y_true)
    print(y_pred)
    print('---\n')

    acc_score = accuracy_score(y_true, y_pred)
    writer.write(' '.join(['ACCURACY SCORE', str(acc_score), '\n']))

    #f1_score = f1_score(y_true, y_pred, average='micro', labels=LABEL_COLUMNS)
    f_score = f1_score(y_true, y_pred, average='micro')
    writer.write(' '.join(['F1-SCORE MICRO', "{:.3f}".format(f_score), '\n']))

    p_score = precision_score(y_true, y_pred, average='micro')
    writer.write(' '.join(['PRECISION', "{:.3f}".format(p_score), '\n']))

    r_score = recall_score(y_true, y_pred, average='micro')
    writer.write(' '.join(['RECALL', "{:.3f}".format(r_score), '\n']))

    f1m_score = f1_score(y_true, y_pred, average='macro')
    writer.write(' '.join(['F1-SCORE MACRO', "{:.3f}".format(f1m_score), '\n\n']))

    # prf_support = precision_recall_fscore_support(y_true, y_pred) # , labels=LABEL_COLUMNS)
    all_p, all_r, all_f, raw = precision_recall_fscore_support(y_true, y_pred) # , labels=LABEL_COLUMNS)
    #print('PRF SUPPORT', prf_support)

    confusion_matrix = multilabel_confusion_matrix(y_true, y_pred) # , labels=LABEL_COLUMNS)
    #print('CONFUSION MATRIX', confusion_matrix)

    for l in range(len(LABEL_COLUMNS)):
        writer.write(LABEL_COLUMNS[l])
        writer.write('\n')
        try:
            writer.write(', '.join(["{:.3f}".format(all_p[l]), "{:.3f}".format(all_r[l]), "{:.3f}".format(all_f[l])]))
            writer.write('\n')
        except:
            writer.write('NONE\n')
        try:
            print(confusion_matrix[l])
            c = [str(f) for f in confusion_matrix[l].tolist()]
            for x in c:
                writer.write(x)
                writer.write('\n')
        except:
            writer.write('NONE\n')
        writer.write('----\n\n')


print('\n THRESHOLDS', l_thresholds)

if args.errors:
    print('## ERRORS')
    for i, e in errors:
        print(i, e)
        print('---')
