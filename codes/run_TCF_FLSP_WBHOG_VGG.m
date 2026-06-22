function results = run_TCF_FLSP_WBHOG_VGG(seq, video_path)
params.gt_boxes = seq.gt_boxes;
params.target_sz = [seq.init_rect(1,4), seq.init_rect(1,3)];    
params.video_path = video_path;  %          

% size, position, frames initialization 
params.init_pos       = [seq.init_rect(1,2), seq.init_rect(1,1)] + floor(params.target_sz/2);  % [y_center, x_center]
params.s_frames       = seq.s_frames;
params.num_frames     = seq.en_frame - seq.st_frame + 1;  
params.seq_st_frame   = seq.st_frame;  
params.seq_en_frame   = seq.en_frame; 

params.newton_iterations   = 5;
params.search_area_shape   = 'proportional';
% Threshold for reducing the cell size in low-resolution cases
params.t_global.cell_selection_thresh = 0.75^2; 

params.feature_downsample_ratio = 4;

params.t_global.cell_size  = 4;  
params.visualization       = 1;         
params.layers = 28;

%% Parameter setting and initialization 
params.filter_max_area  = 50^2; % the size of the training/detection area in feature grid cells
params.search_area_scale = 3;  % [2, 4]
params.scale_exp = [-2, -1, 0, 1, 2];    
params.scale_step = 1.01;   % scale parameters
params.channel = 128;

params.output_sigma_factor = 1/16;   % [1/16,0.1]	
% spatial regularization parameter
params.sigma = 5;
% temporal regularization parameter
params.theta = 15;
% t-CTV regularization parameter
params.lamda = 1e-6; 
% params.rho = 10;
% params.max_mu_t = 1e6;
% penalty parameter
params.penalty_factor = 0.1;   

params.max_iterations = 5;  
% learning rate
params.learning_rate = 0.018;       

%% KF
params.threshold = 0.15;
params.A=[1,0,1,0;0,1,0,1;0,0,1,0;0,0,0,1]; 
params.C=[1,0,0,0;0,1,0,0];  
params.mu=[params.init_pos,0,0]';  %  [y_center, x_center, vy, vx]
params.Sigma=1000*eye(4);  
cQ=1e-7;
cR=1e-2;
params.Q=cQ*[1,0,0,0;0,1,0,0;0,0,1,0;0,0,0,1]; 
params.R=cR*[1,0;0,1]; 

%% run the main function
results = TCF_FLSP_optimized_WBHOG_VGG(params); 
end
