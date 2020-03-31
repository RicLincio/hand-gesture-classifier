function [imleft, imright] = importImages(folder, height, width, m)
%Import multiple images from a folder
%Input: [folder, height, width, m]
%Output: [imleft, imright]
%This function imports the m images, stored inside the folder passed as
%input, into 2 4D arrays of size [height, width, 3, m] where corresponding
%elements in the fourth axis are coupled left and right images.

% create output arrays
imleft = zeros(height, width, 3, m);   imright = imleft;

%import stereo images
for k = 1:m
    filename_l = sprintf('img0%da.png', k);
    filename_r = sprintf('img0%db.png', k);
    filename_l_full = fullfile(folder, filename_l);
    filename_r_full = fullfile(folder, filename_r);
    if exist(filename_l_full, 'file') && exist(filename_r_full, 'file')
        imleft(:,:,:,k) = imread(filename_l_full);
        imright(:,:,:,k) = imread(filename_r_full);
    else
        warningMessage = sprintf('Warning: image file does not exist:\n%s', filename_l_full);
        uiwait(warndlg(warningMessage));
    end
end
imleft = imleft / 255;
imright = imright / 255;

end