% From stereo images compute the disparity map and reconstruct the geometry
% of the scene. Select then the hand points, with the correspondent depth,
% and create an RGBD array.

function out = processStereoImages(imleft, imright, stereoParams)

displayFigures = false;
out = zeros(28, 28, 4, 9);

for i = 1:size(imleft, 4)
    %% image rectification
    I1 = imleft(:,:,:,i); I2 = imright(:,:,:,i);                           % range: [0,1]
    [I1_r, I2_r] = rectifyStereoImages(I1, I2, stereoParams);              % rectified images
    
    % plot
    if displayFigures
        figure; imshowpair(I1, I2,'montage'); title('I1 (left); I2 (right)');
        figure; imshow(stereoAnaglyph(I1_r, I2_r));
    end
    
    %% disparity map computation
    disparityMap.range = [0 800];
    disparityMap.fullMap = disparity(rgb2gray(I1_r), rgb2gray(I2_r), ...   % range: [-3.4028e+38,800]
        'BlockSize', 7, ...
        'DisparityRange', disparityMap.range, ...
        'DistanceThreshold', 2, ...
        'UniquenessThreshold', 1);
    % rectify disparity map
    disparityMap.mask = disparityMap.fullMap >= 0;
    disparityMap.maskedMap = disparityMap.fullMap .* disparityMap.mask;    % range: [0,800]

    % plot
    if displayFigures
        figure; imshow(disparityMap.fullMap, []);                          % range: (-Inf, 799]
        title('Full Disparity Map'); colormap jet; colorbar
        figure; imshow(disparityMap.maskedMap, []);                        % range: [0, 799]
        title('Rectified Disparity Map'); colormap jet; colorbar
        figure; imshow(I1_r .* disparityMap.mask)
    end
    
    %% reconstruct the scene computing the geometric points
    
    coords.D3 = reconstructScene(disparityMap.maskedMap, stereoParams);    % range: (-Inf, Inf)
    % coordinates
    coords.x = coords.D3(:,:,1);
    coords.y = coords.D3(:,:,2);
    coords.z = coords.D3(:,:,3);
    % colors
    coords.r = I1_r(:,:,1);
    coords.g = I1_r(:,:,2);
    coords.b = I1_r(:,:,3);
    % select hand points
    coords.mask = coords.z > 200 & coords.z < 600;
%     coords.mask_smoothed = imgaussfilt(double(coords.mask), 15); % gaussian kernel
%     coords.mask_unsharp = coords.mask + 1 * (coords.mask - coords.mask_smoothed); % unsharp masking
%     coords.mask_mixed = double(coords.mask); + coords.mask_smoothed(coords.mask == 0);
%     coords.mask_mixed(~coords.mask) = coords.mask_smoothed(~coords.mask);

    % plot
    if displayFigures
        figure; imshow(coords.mask); title('Hand points')
    end
    
    %% render final image
    
    % center the image on the hand
	[row, col, ~] = find(coords.mask);
    baricenter = round([sum(row), sum(col)] / numel(row));
    half_edge = floor(size(coords.mask,1) / 2);
    from_col = baricenter(2) - half_edge;
    to_col = baricenter(2) + half_edge;
    
    % resize images of interest and flip them
    depth_temp = disparityMap.maskedMap(:,from_col:to_col) ...             % range: [0,1]
        .* coords.mask(:,from_col:to_col) / (2 * disparityMap.range(2));   % size: 1079 x 1079  
    
    out_big(:,:,1:3,i) = flip(I1_r(:,from_col:to_col,:), 2);               % size: 1079 x 1079, range: [0,1]
    out_big(:,:,4,i) = flip(depth_temp, 2);
    
    % channel 1
    out(:,:,1:3,i) = imresize(out_big(:,:,1:3,i), [28 28], 'lanczos2');
    % channel 2
    out(:,:,4,i) = imresize(out_big(:,:,4,i), [28 28], 'lanczos2');
    
    % plot
    figure; imshowpair(out_big(:,:,1:3,i), out_big(:,:,4,i), 'montage')
end

end