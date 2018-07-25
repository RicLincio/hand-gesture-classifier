I_tot = [imleft(:,:,:,8), imright(:,:,:,8)];
figure; subplot(2,1,1); imshow(I_tot); title('Before correction')
% size(I_tot) == [1080 3840 3]
% I_left from column 1681 to 3600
I_left = I_tot(:,1681:3600,:);
% I_right from column 3601 to 3840 and from 1 to 1680
I_right = zeros(size(I_left));
I_right(:,1:240,:) = I_tot(:,3601:end,:);
I_right(:,241:end,:) = I_tot(:,1:1680,:);
subplot(2,1,2); imshowpair(I_left, I_right, 'montage'); title('After correction')
imleft(:,:,:,8) = I_left;
imright(:,:,:,8) = I_right;
clear I_left I_right I_tot