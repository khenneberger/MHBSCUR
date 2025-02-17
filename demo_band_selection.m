%=========================================================
%
% DEMO BAND SELECTION
% 
% This code runs a demo band selection using the matrix-based HSI Band
% Selection method based on the matrix CUR Decomposition
%
% MATLAB R2023b
% Author: Katherine Henneberger
% Institution: University of Kentucky - Math Department
%  
%=========================================================


clear;clc;close all;
addpath(genpath(pwd))


%%
classifier_names = {'SVM'}; % or 'KNN'
dataset_names = {'Indian_Pines', 'Salinas'};
method_names = {'Proposed'}; % add other methods to compare

svm_para = {'-c 10000.000000 -g 0.500000 -m 500 -t 2 -q', '-c 100 -g 16 -m 500 -t 2 -q'};
%% input parameters
updated_method_ids = 1; % ids of band selection method that you want to perform
update_classifier_ids = {1,1}; % which classifier(s) you want to use in each dataset
K = 30; 
delta = 3;
x = delta : delta : K; % number of the selected bands
num_repe = 1; % repetitive experiments to reduce the randomness of choosing training samples
plot_method_ids = [1, 2, 3, 4]; % indicate which methods to plot
plot_classifier_id = 1; %indicate which classifer to plot

%% initialization
Acc_vals = zeros(num_repe,10,4,2); 
C_cnt = size(classifier_names, 2);
M_cnt = size(method_names, 2);
D_cnt = size(dataset_names, 2);


if ~exist('Methods')
    Methods = cell(1, 1);
    for i = 1 : size(method_names, 2)
        Methods{1, i} = get_method_struct(method_names{i}, dataset_names, classifier_names, K / delta);
    end
end


%% band Selection for each dataset
fidtime = fopen('results\running_time_svm_100iter.txt','w');
for dataset_id = [1,2]
    % load data
    Dataset = get_data(dataset_names{dataset_id});
    Dataset.svm_para = svm_para{1, dataset_id};
    A = Dataset.A;
    X = Dataset.X;
    [M, N, d] = size(A);
    if size(A,3)>1
        [n1,n2,n3] = size(A);
        n = n1*n2;
        A2 = reshape(A,n,n3);
    else
        A2 = A;
        [n,n3] = size(A2);
    end



    %% preprocess
    pretime1 = zeros(10,1);
    
   
     %% Compute the graph Laplacians for MHBSCUR
    tic
    pars.hs = 0.0077; % for Indian Pines find by running spatial/spectral parameters script
    pars.ht = 6.4373; % find by running spatial/spectral parameters script
    pars.Ks = 4;
    pars.Kt = 4;
    [Ls,Lc] = glapv2(A2,pars);
    pretime1(:)= toc;
    % Establish Opts
    opts.tol = 10e-6;
    opts.max_iter = 100;%100;
    opts.beta = 1;
    opts.Ls = Ls;
    opts.Lc = Lc;
    opts.DEBUG = 1;
    %% calculate the band set for each method
    tm1 = pretime1; % cputime for each trial
    
    cnt = 1;
    for j = x % number of bands
             
        opts.k = j;
        opts.rs = round(j*log(n*n3)); % round(rln(mn))=round(20*ln(21025*200))
        opts.cs = round(j*log(n3)); % round(20*ln(200))
        if dataset_id == 1
            load(['parameters\svm_best_pars_indianpines(',num2str(j),').mat'])
            opts.lambda = 100;
            opts.gamma1 = best_pars.gamma1;
            opts.gamma2 = best_pars.gamma2;
            opts.tau = best_pars.tau;
        else
            load(['parameters\svm_best_pars_salinas(',num2str(j),')_.mat'])
            opts.lambda = best_pars.lambda;
            opts.gamma1 = best_pars.gamma1;
            opts.gamma2 = best_pars.gamma2;
            opts.tau = best_pars.tau;
        end
        
        % run proposed method
        fprintf('proposed method ...\n')
        tic;
        [~, bandset, iter] = MHBSCUR(A2, opts); 
        tm1(cnt) = tm1(cnt)+toc;
       
        avgtimeiter = tm1(cnt);
        fprintf(fidtime,  'dataset: %s----k:%d----Method: MHBSCUR----time:%4.4f\n',dataset_names{dataset_id},j,avgtimeiter);
        Methods{1, 1}.band_set{dataset_id, cnt} = bandset;
        cnt = cnt+1;

    end
    
    %% test accuracy

    % Initialization
    for i = updated_method_ids
        for classifier_id = update_classifier_ids{dataset_id}
            for j = 1 : size(x, 2)
                Methods{1, i}.accu(dataset_id, classifier_id, j) = 0;
            end
        end
    end


%% 

     fid = fopen(['results\result_SVM_100iter_',num2str(dataset_id),'_accuracy.txt'],'w');
     for ite = 1 : num_repe
        % refresh the training and testing samples
        if ite > 1
            Dataset = get_data(dataset_names{dataset_id});
            Dataset.svm_para = svm_para{1, dataset_id};
        end
        for classifier_id = update_classifier_ids{dataset_id}
            %% calculate accuracy of each band selection method
            for j = updated_method_ids
                %if j == 1
                %   continue
                %end
                cnt = 1;
                for k = x
                    cur_accu = test_bs_accu(Methods{1, j}.band_set{dataset_id, cnt}, Dataset, classifier_names{classifier_id});
                    Methods{1, j}.accu(dataset_id, classifier_id, cnt) = ...
                        Methods{1, j}.accu(dataset_id, classifier_id, cnt) + cur_accu.OA;
                    Acc_vals(ite,k/3,j,dataset_id) = cur_accu.OA;
                    str = fprintf('ite: %d\t%s----%s----%s----%f\n', ite, dataset_names{dataset_id}, ...
                        classifier_names{classifier_id}, method_names{j}, Methods{1, j}.accu(dataset_id, classifier_id, cnt) / ite);
                    fprintf(fid,'ite: %d\t%s----%s----%s----%f\n', ite, dataset_names{dataset_id}, ...
                        classifier_names{classifier_id}, method_names{j}, Methods{1, j}.accu(dataset_id, classifier_id, cnt) / ite);
                    cnt = cnt + 1;
                end
                fprintf('\n');
            end
        end
    end
    fclose(fid);

    %% calculate the mean accuracy over different iterations
    for classifier_id = update_classifier_ids{dataset_id}
        for j = updated_method_ids
            Methods{1, j}.accu(dataset_id, classifier_id, :) = ...
                Methods{1, j}.accu(dataset_id, classifier_id, :) / num_repe;
        end
    end
save('result_method_SVM.mat','Methods')
end
fclose(fidtime);

%% Save Matrix of Acc Values
save('results\Accuracy_values_SVM.mat',"Acc_vals")

%% Plot result
for i = [1,2]
    fig = plot_method_improved(Methods(1), x, classifier_names, classifier_id, "MHBSCUR", i, 1);
end

%% get a struct of a band selection method

function [method_struct] = get_method_struct(method_name, dataset_names, classifier_names, band_num_cnt)
method_struct.method_name = method_name;
dataset_cnt = size(dataset_names, 2);
classifier_cnt = size(classifier_names, 2);

method_struct.band_set = cell(dataset_cnt, band_num_cnt); % K / delta
method_struct.band_set_corr = cell(dataset_cnt, band_num_cnt);
method_struct.accu = zeros(dataset_cnt, classifier_cnt, band_num_cnt);

end

