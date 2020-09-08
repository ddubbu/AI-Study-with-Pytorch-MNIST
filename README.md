# AI-Study-with-Pytorch-MNIST
MNIST with Pytorch

## Pytorch
1. document  
https://pytorch.org/docs/stable/nn.html#linear-layers

2. 모델 작성의 기초  
https://tutorials.pytorch.kr/beginner/Intro_to_TorchScript_tutorial.html   
    * 설명 
        * forward 함수. 모듈이 호출될 때 실행되는 코드  
      output = model(x)  # 이 자체가 forward 가 되나봐
        * \__init\__ 에서는 layer, forward에서 activation function을 쓰는 이유는?

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

## DNN.py

* MNIST dataset loading 하기  
https://teddylee777.github.io/pytorch/pytorch-mnist-dataloader-loading%ED%95%98%EA%B8%B0

* follow code and model  
https://korchris.github.io/2019/08/23/mnist/

* plot loss and accuracy  
https://m.blog.naver.com/PostView.nhn?blogId=jung2381187&logNo=220408468960&proxyReferer=https:%2F%2Fwww.google.com%2F   
for that, I divide train_**batch**\_loss or train_**epoch**\_loss

* plot mnist result  
https://neurowhai.tistory.com/91