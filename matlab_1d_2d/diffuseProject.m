function [ u1_i ] = diffuseProject(u0,u1,visc,dt)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    fft_u1 = fft2(u1);
    [nx,ny,~] = size(u1);
    for i = 1:nx
        for j = 1:ny
            k = [2*pi*i/nx 2*pi*j/ny];
            fft_u1(i,j,:) = (1/(1+visc*dt*(k*k')))*fft_u1(i,j,:);
            fft_u1(i,j,:) = [fft_u1(i,j,1) fft_u1(i,j,2)]...
                -1/(k*k')*(k*[fft_u1(i,j,1) fft_u1(i,j,2)]')*k;
        end
    end
    u1_i = real(ifft2(fft_u1));
end

