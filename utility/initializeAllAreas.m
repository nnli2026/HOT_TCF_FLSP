function [params, bg_area, fg_area, area_resize_factor] = initializeAllAreas(im, params)
    im_sz = size(im);  % [288 352 3]
    % For objects with large height and width and accounting for at least 10 percent of the whole image,
    % we only search 2x height and width
    if(prod(params.target_sz)/prod(im_sz(1:2)) > 0.05)     % Large target.  From HCF
        params.padding = 1;     
    else
        params.padding = 1.2;                              % Normal target.
    end
	% we want a regular frame surrounding the object我们希望物体周围有一个规则的框架
	params.avg_dim = sum(params.target_sz)/2;  % [95 + 21] / 2 = 58
	% size from which we extract features我们从中提取特征的大小
	bg_area = round(params.target_sz + params.avg_dim * params.padding);  % [95, 21] + 58 * 1.8 = [199, 125]
	% pick a "safe" region smaller than bbox to avoid mislabeling选择一个比bbox小的“安全”区域，以避免贴错标签
	fg_area = round(params.target_sz - params.avg_dim * params.inner_padding); % [83 9]
	% saturate to image size饱和度到图像大小
	if(bg_area(2)>size(im,2)), bg_area(2)=size(im,2)-1; end
	if(bg_area(1)>size(im,1)), bg_area(1)=size(im,1)-1; end
	% make sure the differences are a multiple of 2 (makes things easier later in color histograms)确保差异是2的倍数（这样在颜色直方图中更容易处理）
	bg_area = bg_area - mod(bg_area - params.target_sz, 2);
	fg_area = fg_area + mod(bg_area - fg_area, 2);

	% Compute the rectangle with (or close to) params.fixedArea and same aspect ratio as the target bbox用（或接近）参数计算矩形。fixedArea和与目标bbox相同的宽高比
	area_resize_factor = sqrt(params.fixed_area/prod(bg_area));
	params.norm_bg_area = round(bg_area * area_resize_factor);
	% Correlation Filter (HOG) feature space
	% It smaller that the norm bg area if HOG cell size is > 1  如果HOG细胞大小> 1，则比标准bg面积小
	params.cf_response_size = floor(params.norm_bg_area / params.hog_cell_size);
	% given the norm BG area, which is the corresponding target w and h? 给定范数BG面积，哪个是对应的目标w和h？
 	norm_target_sz_w = 0.75*params.norm_bg_area(2) - 0.25*params.norm_bg_area(1);
 	norm_target_sz_h = 0.75*params.norm_bg_area(1) - 0.25*params.norm_bg_area(2);
    params.norm_target_sz = round([norm_target_sz_h norm_target_sz_w]);
	% norm_delta_area is the number of rectangles that are considered.
	% it is the "sampling space" and the dimension of the final merged resposne
	% it is squared to not privilege any particular direction
	params.norm_delta_area = min(params.norm_bg_area) * [1, 1];

end
