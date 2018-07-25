% Turn a a set of images with 4 channels, containing RGBD information, into
% a similar matrix with only 3 channels, where the wirst one has black and
% white images, the second the relative depth map, while the third is left
% unused.
% Input: a 4D array of size (height x width x 4 x num_images)
% Output: a 3D array of size (height x width x 3 x num_images)

function out = prepareSet(in)

% get input size and shape output
[height, width, ~, num_images] = size(in);
out = zeros([height, width, 3, num_images]);

for i = 1:num_images
    out(:,:,1,i) = rgb2gray(in(:,:,1:3,i));
    out(:,:,2,i) = in(:,:,4,i);
end

end