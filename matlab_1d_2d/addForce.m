function [ u1 ] = addForce( u0_i,F_i,dt)
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
    u1 = u0_i+dt*F_i;
end

