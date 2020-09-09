# AI-Study-with-Pytorch-MNIST
MNIST with Pytorch

## Version
Pytorch - 1.5.1

## After Study
1. 우선, input이 정제되어있고, train, test routine도 만들어 놓으니깐, layer를 추가하고, 실험하는 게 쉬웠다.  
역시 처음부터 코딩을 정갈하게 짜놓아야하는 것 같다.  
2. 하지만, 사소한 lr, dropout 비율 등 상수등을 바꾸는 것은 자동화 실험할 필요성을 느꼈다.  
솔직히 Model class 여러개 짜 놓고, 반복 실험하면 될 것 같단 말이지??  
저장 방법도, 3개 중에 option으로 조절했듯이

## Pytorch
1. document  
https://pytorch.org/docs/stable/nn.html#linear-layers

2. 모델 작성의 기초  
https://tutorials.pytorch.kr/beginner/Intro_to_TorchScript_tutorial.html   
    * 설명 
        * forward 함수. 모듈이 호출될 때 실행되는 코드  
            output = model(x)  # 이 자체가 forward 가 되나봐
        * \__init\__ 에서는 layer, forward에서 activation function을 쓰는 이유는?  
            내 생각인데, init에서 정의한 layer는 여러번 쓸 수 있는거지.  
            forward에서는 그 layer들을 조합해서 쓰는 거고

3. 함수 summary
    <pre><code>from torch summary import summary  
    model = DNN_Net()  
    summary(model, (50, 1, 28, 28)) # with input size
    </code></pre>

4. train_parameter 접근법
    <pre><code>
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        # model.state_dict()[param_tensor].dat => 접근가능하네!
    # 옵티마이저의 state_dict 출력
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])
    </code></pre>

5. 저장하는 방법
https://tutorials.pytorch.kr/beginner/saving_loading_models.html

6. layer 쌓는 법  
아직 예쁘게 쌓는법을 모른다.
    * nn.Sequential
    * nn.ModuleList 
    
    * 특히, __init__에서 정의하는 것, forward에서 정의하는 것
        * argumnet 가 어디에 사용되는지 잘 보시오
    

## DNN.py

1. MNIST dataset loading 하기  
https://teddylee777.github.io/pytorch/pytorch-mnist-dataloader-loading%ED%95%98%EA%B8%B0

2. follow code and model  
https://korchris.github.io/2019/08/23/mnist/

3. plot loss and accuracy  
https://m.blog.naver.com/PostView.nhn?blogId=jung2381187&logNo=220408468960&proxyReferer=https:%2F%2Fwww.google.com%2F   
for that, I divide train_**batch**\_loss or train_**epoch**\_loss

4. plot mnist result  
https://neurowhai.tistory.com/91

## How to use "Drop out"

1. 헷갈렸던 점
    <pre><code> # 언제 아래 두개를 쓰는가?
    torch.nn.Dropout (Python class, in Dropout)
    torch.nn.functional.dropout (Python function, in torch.nn.functional)
   </code></pre>
    
2. 참고한 자료
https://www.it-swarm.dev/ko/neural-network/nn-dropout-vs-f-dropout-pytorch/808111322/

    * nn.Dropout 
        * 이 훈련 중에만 적용되도록 설계됨.
        * 
    * nn.functional.dropout
        * training=False 로 따로 설정해야함.

    | | `nn.Dropout` | `nn.functional.dropout` |
    |:---:|:---|:---|
    | p=버릴 범위 | 훈련 중에만 적용됨 | `training=False` 따로 설정 필요함. |
    | 장 점 | summary에 등록, 레이어 여러번 사용 가능 | |    
    
3. acc(test) > acc(test) 이것은 무슨 상황?  
    * dropout 때문이기는 하지만  
    사실, test accuracy 처음 부터 높았음.
    
    * 도전해볼만한 것  
        * bigger network  
        * train for longer time  
        * advanced optimization  
        * bigger test dataset
        
        
## How to use "Regularization"
1. 방법1 : 직접 weight 모으기  
    <pre><code>
    reg_lambda=0.01
    l2_reg = 0
        if isinstance(layer_names, list):
             for W in self.model.named_parameters():
                 if "weight" in W[0]:
                     layer_name = W[0].replace(".weight", "")
                     if layer_name in layer_names:
                         l2_reg = l2_reg + W[1].norm(2)
    loss = loss + l2_reg * reg_lambda  #  Parameters 가 아닌 Tensor 를 더하면 안됩니다.
    loss.backward()
    </code></pre> 

2. 방법2 : optimizer 함수에서 정의하기 >> 입증된 방법이 아님.
https://lovit.github.io/machine%20learning/pytorch/2018/12/05/pytorch_l1_regularity/

3. Question  
    Q1 .bias는 무시하고 weight 얘만 수식으로 모을까?  
        - 그런데, batch norm weight 도 학습되는거 맞음?  
        - 모든 layer의 weight 를 모으는 거임?  
    Q2. weight tensor가 아닌 param을 모으는 것 같은데, back-propagation 에 적용되는거임??

## How to use "BatchNrom"

1. why Batch_norm before ReLU?
2. learnable parameter ?
3. 적용 효과와 원리
4. drop out 처럼 train, test 달리 적용해야하는 것이 있을까?
https://sy-programmingstudy.tistory.com/10  
테스트시의 배치 정규화

## CNN.py
나의 궁극적인 pytorch 사용 목적이랄까?  
CNN을 편하게 쌓고 싶다. 사실, tf.nn도 sequential 하게 할 수 있잖아.  
굳이 꾸역꾸역 weight tensor 따로 정의하고, 사용했는데,  
그거 말고 MNIST는 28*28 이라서, 쉽게 변화되는 tensor 크기를 알 수 있을거 같은데,

뭔가 내 맘대로 kernel_size, max_pooling size 등을 조절해버리면, 결과는 음수 얘가 나오던데..
N' = (N+2P-F)/S + 1 에서 N' 이라던가, P가 적절치 않다던지 ㅜㅜ


### 확인하고 싶은거
1. input shape이 정사각형꼴이 아닐때, 2D conv를 차원을 계산하면서 사용해야하나?  
2. conv model은 layer마다, 선을 인식, 도형, ..., 고양이를 인식한다는데  
중간 단계를 어떻게 인식함? 

