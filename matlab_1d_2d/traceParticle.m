function [ x0 ] = traceParticle(i,j,x,u0,dt,dx,dy,L)
%UNTITLED9 Summary of this function goes here
%   Detailed explanation goes here
    [nx,ny,~] = size(u0);
    if (i+1)>nx
        ri = 1;
    else 
        ri = i+1;
    end
    if (j+1)>ny
        rj = 1;
    else
        rj = j+1;
    end
    v_x = 1/4*(u0(i,j,:)+u0(i,rj,:)+u0(ri,j,:)+u0(ri,rj,:));
    x0 = x-dt*v_x(:)';
    x0(1) = mod(x0(1),L);
    x0(2) = mod(x0(2),L);
end

