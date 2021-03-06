Ndim = 2;
L = 10;
dt = 0.1;
N = 200;
dx = L/N;
dy = L/N;
visc = 0.00001;
kS = 0; aS = 0 ; Ssource = 0;

%F = 1000*(rand(Ndim,N)-0.5);
F = zeros(N,N,Ndim);
F(20:25,20:25,1)=0.1;
F(20:25,20:25,2)=0.1;
%F = (rand(N,N,Ndim)-0.2);


% vecteur vitesse, u0 �tape n-1, u1 �tape n
u0 = zeros(N,N,Ndim);
u1 = zeros(N,N,Ndim);
x = zeros(N,N,Ndim);
for i = 1:N
    for j = 1:N
        x(i,j,:) = [dx*(i-1+0.5) dy*(j-1+0.5)];
    end
end

% substance transport�e par le flux
s0 = zeros(N,N,Ndim);
s1 = zeros(N,N,Ndim);

i = 1;
figure
while 1
    % handle display and user interaction
    % get forces F and sources Ssource from th UI
    x=mod(x+dt*u1,L);
    %u1(5)
    %x(5)
    %x(12)
    [u0,u1] = swap(u0,u1);
    [s0,s1] = swap(s0,s1);
    u1 = Vstep(u1,u0,visc,F,dt,Ndim,dx,dy,L);
    Sstep(s1,s0,kS,aS,u1,Ssource,dt,Ndim);
    plot(x(:,:,1),x(:,:,2),'.')
    drawnow
    i=i+1;
end