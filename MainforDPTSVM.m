clear; close all; clc;
model = @DPTSVM;      
modelName = func2str(model);
%% =========================================================
% 1. load datasets
%% =========================================================
dataset = "noisyB";  % "noisyA" | "noisyB"
if dataset == "noisyA"
  load("noisyA.mat");
elseif dataset == "noisyB"
  load("noisyB.mat");
end
X = TrainData.X; Y = TrainData.Y;


%% =========================================================
% 2. label preprocess to 1,2,...,nclass
%% =========================================================
[~,~,TrainData.Y] = unique(TrainData.Y);
[~,~,TestData.Y] = unique(TestData.Y);
%% =========================================================
% 3. data normalization
%% =========================================================
[TrainData.X, PS] = mapminmax(TrainData.X');
TrainData.X = TrainData.X';
TestData.X = mapminmax('apply',TestData.X',PS)';

%% =========================================================
% 4. parameter setting for synthetic dataset
%% =========================================================
if dataset == "noisyA"
  Param.c1 = 0.25;        % the parameter for L1-norm within-class term
  Param.c2 = 0.5;        %  the parameter for double-sided hinge loss
elseif dataset == "noisyB"
  Param.c1 = 0.5;        % the parameter for L1-norm within-class term
  Param.c2 = 16;         %  the parameter for double-sided hinge loss
end


%% =========================================================
% 6. training DPTSVM
%% =========================================================
Mld = model(Param);

tic
Mld.train(TrainData);
toc
Res = Mld.test(TrainData);
acc = mean(TrainData.Y==Res.PredictY);
fprintf('Train Accuracy = %.4f\n', acc);

%% =========================================================
% 7. testing on test set
%% =========================================================
Res = Mld.test(TestData);
acc = mean(TestData.Y==Res.PredictY);
fprintf('Test Accuracy = %.4f\n', acc);

