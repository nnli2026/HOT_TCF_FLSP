%% TCF-FLSP-based Tracker for Hyperspectral Object Tracking
clear;clc;close all;
addpath('codes','model','utility','tensor_toolbox_2.5');
addpath 'D:\matlab\toolbox\matconvnet\matlab'
vl_setupnn(); 
global enableGPU;
enableGPU = true;
%% display
disp('--------------TCF-FLSP Tracker-------------')
disp(['Current Time: ',datestr(now)])
curr_time = num2str(fix(clock));
curr_time = strrep(curr_time,' ','');

%% Datasets Setting
base_path = 'D:/dataset/HSV_dataset/validation/HSI-VIS';  %%%% HSI-VIS  HSI-RedNIR  HSI-NIR 

%% HSV Scenarios
videos_vis={'ball';'basketball';'board';'book';'bus';'bus2';'campus';'car';'car2';'car3';'card';'coin';'coke';...
    'drive';'excavator';'face';'face2';'forest';'forest2';'fruit';'hand';'kangaroo';'paper';'pedestrain';...
    'pedestrian2';'player';'playground';'rider1';'rider2';'rubik';'student';'toy1';'toy2';'trucker';'worker'};%35

%% Main 
vid = 1;  
videos = videos_vis;  
disp(['The Scene is: ', videos{vid}])
video_path = [base_path '/' videos{vid}]; 
[seq, ground_truth] = load_video_info(video_path);
seq.VidName = videos{vid};   
st_frame = 1;  
en_frame = seq.len; 
seq.st_frame = st_frame;  
seq.en_frame = en_frame;
seq.gt_boxes = ground_truth;  
%% Run - main function
results  = run_TensorLRRCF_WBHOG_VGG(seq, video_path);

%% success and precision plots
res = results.res;  
%%%%%%%%%%%% 
filename = ['E:/TCF-FLSP_results/vis-' videos{vid} '.txt'];  
dlmwrite(filename, res, 'delimiter', '\t', 'precision', 6); 
distance_precision_threshold=0:50;
PASCAL_threshold=0:0.02:1;
[distance_rec, PASCAL_rec, average_cle_rec]= computeMetric(res,ground_truth,distance_precision_threshold,PASCAL_threshold);
% AUC, DP score
res = [mean(mean(PASCAL_rec(2:51))) ,mean(distance_rec(21))];
display(['AUC:   '   num2str(res(1))   '         DP(20):   '   num2str(res(2))]);
display(['FPS:   '   num2str(results.fps)]);
