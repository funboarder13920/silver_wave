function [ u0,u1 ] = swap( u0,u1 )
    temp = u1;
    u1 = u0;
    u0 = temp;
end

