function [ u1_i] = transport(u1_i,u0_i,u0,dt,dx,dy,L)
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here
    [nx,ny,] = size(u1_i);
    for i=1:nx
        for j = 1:ny
            x = [dx*(i-0.5) dy*(j-0.5)];
            x0 = traceParticle(i,j,x,u0,dt,dx,dy,L);
            u1_i(i,j,1) = linInterp(x0,u0_i,dx,dy,L);
        end
    end
    %u1_i(1,5)
    %u1_i(1,4)
    %u1_i(1,11)
end

