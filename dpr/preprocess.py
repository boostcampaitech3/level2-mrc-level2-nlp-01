import pandas as pd
import pickle
from typing import List
from datasets import load_from_disk, Dataset, DatasetDict

class PreprocessingForDPR:
    def __init__(self, data_path: str, total_length: int):
        self.data_path = data_path
        self.total_length = total_length
        self.original_dataset = load_from_disk(self.data_path)
    
    def make_dataframe_from_dataset(self):
        train_answers = []
        train_context = []
        train_document_id = []
        train_id = []
        train_title = []
        train_question = []

        valid_answers = []
        valid_context = []
        valid_document_id = []
        valid_id = []
        valid_title = []
        valid_question = []

        original_train_dataset = self.original_dataset['train']
        original_valid_dataset = self.original_dataset['validation']

        for train_example in original_train_dataset:
            train_answers.append(train_example['answers'])
            train_context.append(train_example['context'])
            train_document_id.append(train_example['document_id'])
            train_id.append(train_example['id'])
            train_title.append(train_example['title'])
            train_question.append(train_example['question'])
        
        for valid_example in original_valid_dataset:
            valid_answers.append(valid_example['answers'])
            valid_context.append(valid_example['context'])
            valid_document_id.append(valid_example['document_id'])
            valid_id.append(valid_example['id'])
            valid_title.append(valid_example['title'])
            valid_question.append(valid_example['question'])
        
        new_train_context, new_train_answers = self.scale_answers_and_context(train_context, train_answers)
        new_valid_context, new_valid_answers = self.scale_answers_and_context(valid_context, valid_answers)

        self.train_df = pd.DataFrame({'answers' : new_train_answers, 'context' : new_train_context,
                                 'document_id' : train_document_id, 'id' : train_id,
                                 'title' : train_title, 'question' : train_question})
        self.valid_df = pd.DataFrame({'answers' : new_valid_answers, 'context' : new_valid_context,
                                 'document_id' : valid_document_id, 'id' : valid_id,
                                 'title' : valid_title, 'question' : valid_question})
        return self.train_df, self.valid_df
    
    def scale_answers_and_context(self, context: List[str], answers: List[dict]):
        new_context = []
        new_answers = []
        for idx, (example_ctx, example_answer) in enumerate(zip(context, answers)):
            if len(example_ctx) <= self.total_length:
                new_context.append(example_ctx)
                new_answers.append(example_answer)
            else:
                answer_length = len(example_answer['text'][0])
                start_pos = example_answer['answer_start'][0]
                end_pos = start_pos + answer_length
                front_span = start_pos
                back_span = len(example_ctx) - end_pos
                remain_length = self.total_length - answer_length
                front_length = back_length = remain_length // 2

                if front_length > front_span:
                    back_length += front_length - front_span
                    front_length = front_span
                elif back_length > back_span:
                    front_length += back_length - back_span
                    back_length = back_span
                
                new_context.append(example_ctx[start_pos - front_length:end_pos + back_length])
                new_answers.append({'answer_start' : [front_length], 'text' : example_answer['text']})

                assert example_answer['text'][0] == new_context[idx][front_length:front_length + len(example_answer['text'][0])]

        return new_context, new_answers

    def convert_Df_to_HfDataset(self):
        train_dataset = Dataset.from_pandas(self.train_df)
        valid_dataset = Dataset.from_pandas(self.valid_df)
        return train_dataset, valid_dataset

if __name__ == '__main__':
    data_path = '/opt/ml/input/data/train_dataset'
    preprocess = PreprocessingForDPR(data_path, 600)
    preprocess.make_dataframe_from_dataset()
    train_dataset, valid_dataset = preprocess.convert_Df_to_HfDataset()
    datasets = DatasetDict({'train' : train_dataset, 'validation' : valid_dataset})
    print(type(train_dataset))
    print(type(valid_dataset))
    print(type(datasets))
    print(datasets)
    with open('./dataset_dpr.pkl', 'wb') as f:
        pickle.dump(datasets, f)
    with open('./dataset_dpr.pkl', 'rb') as f:
        d = pickle.load(f)
    print('Finished!')
    print(d)