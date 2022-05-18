![image](https://user-images.githubusercontent.com/82494506/168751972-956f972a-c4d9-45c5-84e1-edd5bd9d1279.png)

[3등 솔루션 공유 발표자료](https://github.com/boostcampaitech3/level2-mrc-level2-nlp-01/blob/main/assets/MRC%203%EB%93%B1%20%EC%86%94%EB%A3%A8%EC%85%98%20%EA%B3%B5%EC%9C%A0.pptx.pdf)

## Index
1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Solutions](#3-solutions)
4. [Results](#4-results)
5. [Usages](#5-usages)
6. [Contributors](#6-contributors)

## 1. Project Overview
### 소개
- Retriever Task와 Reader Task를 구성하고 통합하여, 질문을 던졌을 때 답변을 해주는 ODQA 시스템 개발
- Retriever
    - 방대한 Open Domain Dataset에서 질의에 알맞은 지문을 찾아오는 Task
- Machine Reading Comprehension(MRC)
    - 지문이 주어진 상황에서 질의에 대해 응답하는 기계 독해 Task
- Open-Domain Question Answering(ODQA)
    - Retriever 와 MRC Task를 결합한 시스템
- P stage 3 대회를 위한 베이스라인

### 평가 방법
#### EM(Exact Match)
![image](https://user-images.githubusercontent.com/82494506/168542423-c81a5595-ab68-4b6d-b811-1ab53857ada5.png)
#### F1 score
![image](https://user-images.githubusercontent.com/82494506/168542194-ae09fc31-e487-4efa-8e51-6eab2374b2b4.png)

## 2. Architecture
### 파일 구성
#### 저장소 구조

```bash
.
├── README.md
├── arguments.py
├── assets                                # readme 에 필요한 이미지 저장
├── config                                # retriever config directory
├── dpr                                   # Dense Passage Retriever module 
├── inference.py                          # 대회 베이스 라인 ODQA 모델 평가 또는 제출 파일 (predictions.json) 생성 
├── inference_elasticsearch.py            # elasticsearch ODQA 모델 평가 또는 제출 파일 (predictions.json) 생성 
├── inference_sparse.py                   # sparseretriever ODQA 모델 평가 또는 제출 파일 (predictions.json) 생성 
├── install                               # 요구사항 설치 파일 
├── jupyternote_books                     # experimental jupyter notebook directory 
├── requirements.txt
├── retrieval.py                          # baseline - sparse retriever 
├── retrieval_elasticsearch.py            # elasticsearch retriever  모듈 평가 및 실행 
├── retrieval_elasticsearch_setup.py      # elasticsearch set-up 파일 
├── retrieval_sparse.py                   # sparse retriever 모듈 평가 및 실행  
├── retriever                             # retriever 모듈
├── retriever_result                      # retriever result directory 
├── train.py                              # 대회 베이스라인 MRC, Retrieval 모델 학습 및 평가 
├── train_data_aug.py                     # MRC, Retrieval 모델 학습 및 평가 
├── trainer_qa.py                         # MRC 모델 학습에 필요한 trainer 제공
├── utils                                 # 유틸함수 모듈 
└── utils_qa.py                           # train_qa 기타 유틸 함수 제공 

```
### 데이터 소개

아래는 대회에서 사용한 데이터셋의 구성을 보여줍니다.

![데이터셋](https://user-images.githubusercontent.com/38339347/168778410-a1a23406-16ef-4d7d-b49e-e94f09267448.png)


MRC 대회에서 기본적으로 제공한 데이터셋은 편의성을 위해 Huggingface 에서 제공하는 datasets를 이용하여 pyarrow 형식의 데이터로 저장되어 있습니다. 다음은 `./data` 구조입니다.
```bash
./data/                        # 전체 데이터
    ./train_dataset/           # 학습에 사용할 데이터셋. train 과 validation 으로 구성 
    ./test_dataset/            # 제출에 사용될 데이터셋. validation 으로 구성 
    ./wikipedia_documents.json # 위키피디아 문서 집합. retrieval을 위해 쓰이는 corpus.
```

data에 대한 argument 는 `arguments.py` 의 `DataTrainingArguments` 에서 확인 가능합니다. 

본 대회에서 사용한 외부데이터셋은 아래 링크에서 다운받을 수 있습니다. 
- [ko_wiki_v1_squad](https://aihub.or.kr/aidata/84)
- [KorQuAD v1.0](https://korquad.github.io/KorQuad%201.0/)

## 3. Solutions
### **Retriever**

#### **Spasrse retriever**
- TF-IDF, BM25 등 스코어 함수들을 사용하여 Sparse retriever를 구현했습니다.
- 'klue/roberta-large', 'klue/roberta-large', 'monologg/koelectra-base-v3-discriminator', 'Mecab' 와 같은 tokenizer 와 스코어 함수들의 조합을 실험했습니다. 
- Elasticsearch를 사용하여 (nori-tokenizer, BM25) Sparse retriever를 구현했습니다. 

#### **Data Preprocessing**
- DPR 모델에 입력으로 들어가기 위하여 Data 중 Context Data에 preprocessing을 하였습니다.
    - Sparse Retrieval의 TF-IDF 및 BM25에서는 문장의 길이에 제한이 없는 반면에, DPR에서 encoder에 사용되는 klue/roberta-large 모델의 경우, 최대 입력받을 수 있는 문장의 길이가 512임. 이에 따라 문장의 길이를 줄여주었습니다.
    - Context 내에 answer가 있는 부분을 기준으로 문장 내의 글자 개수가 최대 600개가 되게 문장이 잘라지도록 하였습니다.

#### **DPR retriever**
- In-Batch Negative: question과 positive sentence로 이루어진 mini-batch 내부에서 다른 example들 사이에서 내적을 통하여 Prediction Score Matrix를 구했습니다.
- Batch-size는 8로 하여 훈련을 진행하였음. 즉, 질문 1개 당 8개의 문장 중 positive sentence 1개를 찾도록 훈련되었습니다.

### **Reader**
#### **Main Model 선정**
baseline code에서 'klue/bert-base', 'klue/roberta-large', 'xlm-roberta-large', 'xlnet-large-cased'로 모델만 바꾸어 성능을 측정했습니다.
이 중 'klue/roberta-large'가 EM 39.5800으로 가장 높은 성능을 보여 해당 모델을 main model로 선정했습니다.

#### **Data Augmentation**
주어진 4천여개(train set 기준)의 데이터로는 다양한 context, question에 대응하기가 어려울 것이라 판단, 외부 데이터를 사용해 데이터를 증강했습니다. (본 대회는 외부 데이터 허용)
- [ko_wiki_v1_squad](https://aihub.or.kr/aidata/84)
    - AI HUB의 '일반상식' 데이터셋 중 'wiki 본문에 대한 질문-답 쌍'
    - train set 기준 약 6만 개
- [KorQuAD v1.0](https://korquad.github.io/KorQuad%201.0/)
    - The Korean Quesiton Answering Dataset
    - train set 기준 약 6만 개

#### **Hyper-parameters tuning**
lr rate, warmup ratio, epochs, batch size 등 hyper parameter를 바꾸어가며 실험했습니다.
이 중 **batch size 변경**이 가장 효과적이었으며 'klue/roberta-large', **batch size 128**일 때 가장 높은 성능을 보였습니다.

#### **Ensemble**
Retriever, Reader 각각의 고도화를 마친 후 통합하여 inference를 진행하였을 때 최고 성능은 **EM 63.3300**이었습니다.
하지만 validation set으로 확인해봤을 때 특정 question에서 자주 예측을 실패하는 것을 확인했습니다.
단일 모델 고도화로는 한계가 있다고 생각하여 다양한 모델을 통한 Ensemble을 진행했습니다.

`soft_voting.py` 파일을 통해 soft-voting 진행 가능합니다. nbest_prediction.json의 예측 텍스트와 그 확률을 활용하여 각 모델이 얼만큼의 확신을 가지고 있는지, 특정 텍스트가 얼마나 많은 모델에서 예측된 텍스트인지를 반영하도록 했습니다.

최종적으로 다음의 두가지 앙상블을 사용하여 얻은 예측들을 제출하였습니다. 
1. 'klue/roberta-large' 모델 4개
2. 'klue/roberta-large' 모델 4개, 'ko-electra-base' 1개, 'xlm-roberta-large' 1개

이와 관련하여 다음과 같은 질문이 있을 수 있습니다.
- '2'의 경우, 왜 각 모델을 동일한 개수만큼 앙상블하지 않았나?
    - `soft_voting.py`에서 구현한 방식의 특성 상 각 모델을 동일한 개수로 앙상블할 경우 **'특정 모델이 오답에 대해 확신을 가질 경우 앙상블된 최종 예측도 오답이 될 가능성이 높다'** 는 단점이 있습니다.
    - 'klue/roberta-large'와 타 모델 간의 성능 격차가 컸기 때문에 정답을 맞출 확률이 높은 'klue/roberta-large'에 가중치를 주기로 했습니다.

## 4. Results
### Retriever
BM25+Mecab → Accuracy : 0.9365  
ElasticSearch → Accuracy : 0.9503

### Reader
#### **Data Augmentation**
EM 39.5800 → 45.8300

#### **Hyper-parameters tuning**
EM 48.7500 → 58.3300

### **Ensemble(soft-voting)**
1. 'klue/roberta-large' 4개</br>(public test set) EM 62.9200, F1 75.7400 → (private test set) **EM 64.4400**, F1 77.0300
2. 'klue/roberta-large' 4개 + 'ko-electra' 1개 + 'xlm-roberta-large' 1개</br>(public test set) EM 62.0800, F1 75.0600 → (private test set) **EM 64.4400**, F1 76.9100

### 최종 결과
#### Public Dataset -> 7등
![image](https://user-images.githubusercontent.com/82494506/168751336-df7317db-4b3e-4357-9d98-9d331556c407.png)

#### Private Dataset -> 3등
![image](https://user-images.githubusercontent.com/82494506/168751216-7a965199-768c-456a-9327-59f80a46647f.png)

## 5. Usages
### 설치 방법

#### 요구 사항

```
# data (51.2 MB)
tar -xzf data.tar.gz

# 필요한 파이썬 패키지 설치. 
bash ./install/install_requirements.sh
```
### Retriever

#### train/evaluate Sparse Retriever
```./config/retrieval_config.json```을 기준으로 Sparse Retriever 를 생성 및 평가합니다.  
```python
# Sparse Retriever 생성 및 평가  
# retriever configuration : ./config/retrieval_config.json 에 따라 
# 저장 경로 : './retriver_results/{MODELNAME}' 
python retrieval_sparse.py --config_retriever ./config/retrieval_config.json
```

#### train/evaluate Elasticsearch Retriever
```./config/elasticsearch_config.json```을 기준으로 Elasticsearch index 를 구축합니다.  
이후 구축한 Elasticsearch index 기준으로 문서를 추출 및 평가합니다. 
```python
# elasticsearch set-up
python retrieval_elasticsearch_setup.py --config_elasticsearch ./config/elasticsearch_config.json

# elasticsearch 평가 
python retrieval_elasticsearch.py --index_name wikipedia_documents \ # elasticsearch_config.json 의 index_name
        --context_path wikipedia_documents.json \
        --output_path ./retriever_result \
        --dataset_name ./data/train_dataset \
        --top_k 20
```

### Reader
#### train

만약 arguments 에 대한 세팅을 직접하고 싶다면 `arguments.py` 를 참고해주세요. 

roberta 모델을 사용할 경우 tokenizer 사용시 아래 함수의 옵션을 수정해야합니다.
베이스라인은 klue/bert-base로 진행되니 이 부분의 주석을 해제하여 사용해주세요 ! 
tokenizer는 train, validation (train.py), test(inference.py) 전처리를 위해 호출되어 사용됩니다.
(tokenizer의 return_token_type_ids=False로 설정해주어야 함)

```python
# train_data_aug.py
def prepare_train_features(examples):
        # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
        # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            # return_token_type_ids=False, # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
            padding="max_length" if data_args.pad_to_max_length else False,
        )
```

기본 제공된 데이터셋 외에 추가 데이터셋을 사용하고 싶은 경우 'ko_wiki', 'korquad' arguments를 사용할 수 있음.
```bash
# 학습 예시
# 기본 데이터셋만 사용하고 싶은 경우
python train_data_aug.py --output_dir ./models/train_dataset --do_train

# 기본 데이터셋에 ko_wiki_v1_squad만 추가하여 사용하고 싶은 경우
python trian_data_aug.py --output_dir ./models/train_dataset --do_train --ko_wiki

# 기본 데이터셋에 korquad만 추가하여 사용하고 싶은 경우
python train_data_aug.py --output_dir ./models/train_dataset --do_train --korquad

# 기본 데이터셋에 ko_wiki_v1_sqaud, korquad를 모두 추가하여 사용하고 싶은 경우
python train_data_aug.py --output_dir ./models/train_dataset --do_train --ko_wiki --korquad
```

#### eval

MRC 모델의 평가는(`--do_eval`) 따로 설정해야 합니다.  위 학습 예시에 단순히 `--do_eval` 을 추가로 입력해서 훈련 및 평가를 동시에 진행할 수도 있습니다.

```bash
# mrc 모델 평가 (train_dataset 사용)
python train_data_aug.py --output_dir ./outputs/train_dataset --model_name_or_path ./models/train_dataset/ --do_eval 
```

### Retriever + Reader

#### eval (Sparse Retriever/Elasticsearch Retriever + MRC)
retriever 와 mrc 모델로 end-to-end 평가를 합니다.
```python
# Sparse Retriever + MRC - train data 평가
python inference_sparse.py --output_dir ./result \
        --dataset_name ../data/train_dataset \
        --model_name_or_path {MRC_MODEL_PATH} \
        --retriever_path {RETRIEVER_MODEL_PATH} \
        --do_eval \
        --top_k_retrieval 20 \
        --overwrite_output_dir
        --answer_postprocessing False \
        --per_device_eval_batch_size 32 \
        --max_seq_length 512

# Elasticsearch + MRC 평가 - train data 평가
python inference_elasticsearch.py --output_dir ./result \
        --dataset_name ../data/train_dataset \
        --model_name_or_path {MRC_MODEL_PATH} \
        --retriever_path Elasticsearch \
        --index_name wikipedia_documents \
        --do_eval \
        --top_k_retrieval 30 \
        --overwrite_output_dir \
        --answer_postprocessing False \
        --per_device_eval_batch_size 32 \
        --max_seq_length 512
```

#### inference (Sparse Retriever/Elasticsearch Retriever + MRC)
retriever 와 mrc 모델로 end-to-end test data에 대해 inference를 합니다.
```python
# Sparse Retriever + MRC 평가 - test data inference
python inference_sparse.py --output_dir ./result \
       --dataset_name ../data/test_dataset \
       --model_name_or_path {MRC_MODEL_PATH} \
       --retriever_path {RETRIEVER_MODEL_PATH} \
       --do_predict \
       --top_k_retrieval 30 \
       --overwrite_output_dir \
       --answer_postprocessing False \
       --per_device_eval_batch_size 32 \
       --max_seq_length 512

# Elasticsearch + MRC 평가 - test data inference
python inference_elasticsearch.py --output_dir ./result \
        --dataset_name ../data/test_dataset \
        --model_name_or_path {MRC_MODEL_PATH} \
        --retriever_path Elasticsearch \
        --index_name wikipedia_documents \
        --do_predict \
        --top_k_retrieval 30 \
        --overwrite_output_dir \
        --answer_postprocessing False \
        --per_device_eval_batch_size 32 \
        --max_seq_length 512
```

#### Ensemble(soft-voting)
`soft_voting.py`와 arguments를 이용하여 ensemble을 진행할 수 있습니다. 각 모델 inference 시 생성되는 `nbest_predictions.json`(not `predictions.json`)을 모아놓은 디렉토리를 `--cand_dir` argument로 입력하면 됩니다.
```python
# 앙상블 예시
python soft_voting.py --cand_dir ./ensemble
# '--description'로 원하는 경우 앙상블에 대한 간략한 설명 추가 가능
```

### Things to know

1. `train_data_aug.py` 에서 sparse embedding 을 훈련하고 저장하는 과정은 시간이 오래 걸리지 않아 따로 argument 의 default 가 True로 설정되어 있습니다. 실행 후 sparse_embedding.bin 과 tfidfv.bin 이 저장이 됩니다. **만약 sparse retrieval 관련 코드를 수정한다면, 꼭 두 파일을 지우고 다시 실행해주세요!** 안그러면 기존 파일이 load 됩니다.

2. 모델의 경우 `--overwrite_cache` 를 추가하지 않으면 같은 폴더에 저장되지 않습니다. 

3. `./outputs/` 폴더 또한 `--overwrite_output_dir` 을 추가하지 않으면 같은 폴더에 저장되지 않습니다.

## 6. Contributors
- 강범서_T3002 : https://github.com/Kang-Beom-Seo
- 오필훈_T3127 : https://github.com/philhoonoh
- 이예진_T3158 : https://github.com/leeyejin1231
- 한기백_T3232 : https://github.com/ivorrr987
- 정유리_T3242 : https://github.com/hummingeel

