function [ u1_i ] = diffuseProject(u0_i,u1_i,visc,dt)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    fft_u1_i = fft(u1_i); 
    k = zeros(fft_u1_i);
    for i = 1:length(k)
       k(1,i) = i; 
    end
    fft_u1_i = (1/(1+visc*dt*(k.*k')))*fft_u1_i;
    fft_u1_i = fft_u1_i - 1/(k*k')*(k.*fft_u1_i')*k; 
    u1_i = real(ifft(fft_u1_i));
end

