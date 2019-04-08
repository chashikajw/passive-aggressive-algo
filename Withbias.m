Traindata = csvread('datafile.csv');
Testdata = csvread('testData.csv');
C = 1;
Iterations = [1,2,10];
AlgoNo = "PA";

function weights = PassiveAggresiveTraining (trainingData, C,AlgoNo, MaxIter)
  X_T = trainingData(:,2:10);
  N = size(X_T,1);
  X_T = [X_T,ones(N,1)]; %add bias column
  Y = trainingData(:,11); %classes
  d = size(X_T,2); 
  weights = zeros(d,1);
  X = X_T'; 
  
  % train process
  for i=1:MaxIter
    for n = 1:N
      x = X(:,n);
      
      y_predict = weights'*x;
      y = getCLass(Y(n,1));
      
      l = max(0,1- y*(weights'*x)); %only sign is wrong it return a nonzero value. otherwise its 0 
      tau = 0;
        switch (AlgoNo)
          case "PA"
            tau = l/ (norm(x,2)^2);           
          case "PA-I"
            tm = l/ (norm(x,2)^2);
            tau =  min (C, tm);          
          case "PA-II"
            tau = l/ ((norm(x,2)^2) + (1/2*C));        
          otherwise
            tau = l/ ((norm(x,2)^2) + (1/2*C));
        endswitch
               
      weights = weights + tau*y*x;  %weight update
        
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
function predictions = TestAccuracy(testdata,w)
  Test_T = testdata(:,2:10);
  N = size(Test_T,1);
  Test_T = [Test_T,ones(N,1)]; %add bias column
  Yclass =testdata(:,11);
  N =size(Test_T,1);
  Test = Test_T';
  correct = 0;
  incorrect = 0;
  
  for n = 1:N
    xt = Test(:,n);
    y_predict = w'*xt;
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

