Traindata = csvread('datafile.csv');
Testdata = csvread('testData.csv');
C = 1;
Iterations = [1,2,10];
AlgoNo = "PA";

function weights_bias = PassiveAggresiveTraining (trainingData, C,AlgoNo, MaxIter)
  X_T = trainingData(:,2:10);
  Y = trainingData(:,11); %classes
  N = size(X_T,1);
  d = size(X_T,2); 
  weights = zeros(d,1);
  bias = 1;
  weights_bias = [weights;bias];
  X = X_T'; 
  
  % train process
  for i=1:MaxIter
    for n = 1:N
      x = X(:,n);
      bias_x = 1;
      x_bias = [x;bias_x];
      y_predict = weights_bias'*x_bias;
      y = getCLass(Y(n,1));
      
      l = max(0,1- y*(weights_bias'*x_bias)); %only sign is wrong it return a nonzero value. otherwise its 0 
      tau = 0;
        switch (AlgoNo)
          case "PA"
            tau = l/ (norm(x_bias,2)^2);           
          case "PA-I"
            tm = l/ (norm(x_bias,2)^2);
            tau =  min (C, tm);          
          case "PA-II"
            tau = l/ ((norm(x_bias,2)^2) + (1/2*C));        
          otherwise
            tau = l/ ((norm(x_bias,2)^2) + (1/2*C));
        endswitch
               
      weights_bias = weights_bias + tau*y*x_bias;  %weight update
        
    endfor
  endfor
  disp(AlgoNo);
  printf("No of iterations: %d\n", MaxIter)
  
  return;
  
endfunction

%convert 2,4 class to -1 and 1
function class = getCLass (y)
  class = -1;
  if (y == 2)
    return;
  else
    class = 1;
    return;
  endif
endfunction


%accuracy test function
function TestAccuracy(testdata,w)
  Test_T = testdata(:,2:10);
  Yclass =testdata(:,11);
  N =size(Test_T,1);
  Test = Test_T';
  correct = 0;
  incorrect = 0;
  
  for n = 1:N
    xt = Test(:,n);
    xt_bias = [xt;1];
    y_predict = w'*xt_bias;
    y = getCLass(Yclass(n,1));
    
    if(y_predict*y<=0)
      incorrect  += 1;
    else
      correct += 1;
    endif  
  endfor
  
  accuracy = (correct/N)*100;
  
  printf("totalData: %d\n", N)
  printf("Incorrect predictions: %d\n", incorrect)
  printf("Correct predictions: %d\n", correct)
  printf("Accuracy: %d\n", accuracy)
  printf("\n")
  
endfunction


for MaxIter = Iterations
  updatedWeights = PassiveAggresiveTraining (Traindata, C,AlgoNo, MaxIter);
  TestAccuracy(Testdata,updatedWeights);
endfor

