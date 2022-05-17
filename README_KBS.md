2. [Solution](#2.-Solution)
3. [Result](#3.-Result)

## 2. Solution(KBS)

#### Data Preprocessing(KBS)
- DPR 모델에 입력으로 들어가기 위하여 Data 중 Context Data에 preprocessing을 하였음.
    - Sparse Retrieval의 TF-IDF 및 BM25에서는 문장의 길이에 제한이 없는 반면에, DPR에서 encoder에 사용되는 klue/roberta-large 모델의 경우, 최대 입력받을 수 있는 문장의 길이가 512임. 이에 따라 문장의 길이를 줄여주었음.
    - Context 내에 answer가 있는 부분을 기준으로 문장 내의 글자 개수가 최대 600개가 되게 문장이 잘라지도록 하였음.

#### Retriver
 - DPR retriever(KBS)
    - In-Batch Negative: question과 positive sentence로 이루어진 mini-batch 내부에서 다른 example들 사이에서 내적을 통하여 Prediction Score Matrix를 구하였음.
    - Batch-size는 8로 하여 훈련을 진행하였음. 즉, 질문 1개 당 8개의 문장 중 positive sentence 1개를 찾도록 훈련되었음.


## 4. Result
#### Retriver 결과(KBS)
- Training 시에는 97% 정도의 정확도를 달성하였음.(Batch-size=8)
- 그러나, 전체 훈련 데이터(약 3700개 기준) 기준 평가 시 Top-20에서 79% 정도의 정확도를 달성하였음.