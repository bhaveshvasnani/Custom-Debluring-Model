% video input

% vid = videoinput('winvideo', 1);
% set(vid, 'ReturnedColorSpace', 'RGB');
% I = im2double(getsnapshot(vid));
% figure, imshow(I)
 
%input image

I1=imread('C:\Users\User\Downloads\image\g.jpg');
I = im2double(imread('C:\Users\User\Downloads\image\g.jpg'));
figure, imshow(I)
title('Input Image');
 
% simulate blur

LEN = 21;
THETA = 11;
PSF = fspecial('motion', LEN, THETA);
blurred = imfilter(I, PSF, 'conv', 'circular');
figure, imshow(blurred);
title('Blurred Image');
 
% add gaussian noise to the image

noise_mean = 0;
noise_var = 0.0001;
blurred_noisy = imnoise(blurred, 'gaussian', noise_mean, noise_var);
figure, imshow(blurred_noisy)
title('Blurred and Noisy (Gaussian) Image')

% adding poisson noise to the image

blurred_noisy = imnoise(blurred_noisy,'poisson')
blurred_noisy = imnoise(blurred, 'gaussian', noise_mean, noise_var);
figure, imshow(blurred_noisy)
title('Blurred and Noisy (Gaussian + Poisson) Image')

% customised guassian deblurring

im=rgb2gray(blurred);
fc=100;
imf= fftshift(fft2(im));
[co,ro]=size(im);
out = zeros(co,ro);
cx = round(co/2);
cy = round(ro/2);
H = zeros(co,ro);

for i = 1 : co
    for j = 1 : ro
        d = (i-cx).^2 + (j-cy).^2;
        H(i,j) = exp(-d/2/fc/fc);
    end;
end;
outf= imf.*H;
out=abs(ifft2(outf));
figure, imshow(out);
 
% deblur

estimated_nsr = noise_var / var(out(:));
J = deconvwnr(blurred_noisy, PSF, estimated_nsr);
figure, imshow(J)
 
% second pass

V = .002;
PSF = fspecial('gaussian',5,5);
luc3 = deconvlucy(J,PSF,15);
figure, imshow(luc3)
title('Restored Image with Damping, NUMIT = 15');
 
% remove noise

H = fspecial('gaussian',2, 5);
J = imfilter(J, H);
figure, imshow(J)
 
% harmonic mean

dim=3;
m = harmmean(J);
harmmean(J,dim);
figure, imshow(J)
title('Harmonic mean')
 
stretched_truecolor = imadjust(J,stretchlim(J));
figure, imshow(stretched_truecolor)
title('Truecolor Composite after Contrast Stretch')

% sharpen

H = padarray(2,[2 2]) - fspecial('gaussian' ,[5 5],2);
sharpened = imfilter(stretched_truecolor,H);
figure, imshow(sharpened);
title('Sharpened image')

% % psnr comparision
% 
% [peaksnr, snr] = psnr(I1, sharpened);
% fprintf('\n The Peak-SNR value is %0.4f', peaksnr);
% fprintf('\n The SNR value is %0.4f \n', snr);

% image comparision
% 
% i3 = xor(sharpened, I1);
% imagesc(i3)
% d = sum(i3(:)) / numel(i3)


% %SSIM

% [ssimval, ssimmap] = ssim(sharpened,I1);
% fprintf('The SSIM value is %0.4f.\n',ssimval);
% figure, imshow(ssimmap,[]);
% title(sprintf('ssim Index Map - Mean ssim Value is %0.4f',ssimval));