function [u1] = Vstep(u1,u0,visc,F,dt,Ndim,dx,dy,L)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
for i=1:Ndim
    u1(:,:,i) = addForce(u0(:,:,i),F(:,:,i),dt);
end
for i=1:Ndim
    u0 = u1;
    %u1(1,5)
    u1(:,:,1) = transport(u1(:,:,i),u0(:,:,i),u0,dt,dx,dy,L);
    %u1(2)
    %u0(2)
end
    u0 = u1;
    u1 = diffuseProject(u0,u1,visc,dt);

