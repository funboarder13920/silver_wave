function [ u1_i ] = linInterp(x0,u0,dx,dy,L)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    inv_x0 = ([1/dx 1/dy].*x0)+0.5;
    i_x0 = fix(inv_x0);
    r_x0 = max(0,inv_x0 - double(i_x0));
    [nx,ny,~] = size(u0);
    if i_x0(1)+1 > nx
        ri = 1;
    else
        ri = i_x0(1)+1;
    end
    if i_x0(2)+1 > ny
        rj = 1;
    else
        rj = i_x0(2)+1;
    end
    i_x0(1) = 1+mod(i_x0(1)-1,L*dx);
    i_x0(2) = 1+mod(i_x0(2)-1,L*dy);
    u1_i = (1-r_x0(2))*(1-r_x0(1))*u0(i_x0(1),i_x0(2),:) +(1-r_x0(2))*(r_x0(1))*u0(ri,i_x0(2),:)...
    +(r_x0(2))*(1-r_x0(1))*u0(i_x0(1),rj,:)+(r_x0(2))*(r_x0(1))*u0(ri,rj,:);
end

