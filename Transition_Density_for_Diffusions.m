%% 1. Hermite function
function [temp]=Hermite(k)
% Probabilist version
% This function computes the Hermite Polynomial Recursively
% Note Hermite(1)=1
 
syms z
 
H{1}=sym('1');
% H{1}=simplify(z*H0-diff(H0,z));
for n=2:k
    H{n}=simplify(z*H{n-1}-diff(H{n-1},z));
end
temp=H{k};
end


%% 2. Obtaining transition density for SDE

%%%% Variables in X(t), Y(t), Z(t) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%global xs ys zs pX pY pZ 
syms a  b  c
syms xs ys zs
syms x  y  z
syms h t s 
 
 
K=4;
J=5;
 
%%%%% Drift and Diffusion for X(t) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
% Vasicek%%%%%%%%%%%%%%%%%%%%%%%%%
 
muX=a*(b-x);
sigmaX=c;
 
%sigmaX=c*sqrt(x);
 
% Black Scholes%%%%%%%%%%%%%%%%%%%%
% muX=a*x;
% sigmaX=b*x;
 
%%%%% Transformation X(t) to Y(t)  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
fX2Y=int(1/sigmaX,x);
%  simplify(finverse(fX2Y))
 
 
fY2X=subs((finverse(fX2Y)), x,y);
 
%%%%%  Drift and Diffusion for Y(t) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
muY_temp=muX/sigmaX-sym('1')/sym('2')*diff(sigmaX,x,1);
muY=subs(muY_temp, x, fY2X);
muY=simplify(muY);
 
sigmaY=sym('1');
 
 
%%%%%%  Transformation Y(t) to Z(t) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
fY2Z=h^(-1/2)*(y-ys);
%   subs(fY2Z, y,ys)
 
fZ2Y=h^(1/2)*z+ys;
%   subs(fZ2Y, z,zs)
 
 
%%%%% Generating Beta   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
 
syms Htemp Expectation Beta  
clear Beta Htemp Expectation  
 
for n=1:K
     HTemp=subs(Hermite(n), z, fY2Z);
     Expectation=HTemp;
 
     for k=1:J 
       HTemp=muY*diff(HTemp,y,1)+sym('1')/sym('2')*diff(HTemp, y, 2);
       Expectation=Expectation + h^k/factorial(k)*HTemp;
     end
     Beta{n}= sym('1')/factorial(n-1) * subs(Expectation, y, ys);
end
 
%%%%% Generating pZ With Loop %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
pZ=sym('0');
 
for m=1:K
  pZ=pZ+Beta{m}*Hermite(m);
end
symvar(pZ);
 
%%%%% Generating pY pX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
pZ=exp(-z^2/2)/sqrt(2*pi)*pZ;
pY=(h^(-1/2))*subs(pZ, z, fY2Z);
pX=(sigmaX^(-1))*subs(pY, y, fX2Y);
pX=subs(pX, ys, subs(fX2Y, x, xs));
pX=simplify(pX);
 
 
%%%%% ploting  pX   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
g1=subs(pX, {a,b,c,h,xs}, {1,1,2,1/250,1});
figure(1)
ezplot(g1)
%ezplot(log(g1), [0, 2])
 
 
%%%%%% Ploting EXact Density for Vasicek %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
gamm=sigmaX*sqrt(1-exp(-2*a*h));
 
density=(pi*gamm^2/a)^(-1/2)*exp(-(x-b-(xs-b)*exp(-a*h))^2*a/(gamm^2));
g2=subs(density, {a,b,c,h,xs},{1,1,2,1/250,1});
g2=simplify(g2);
figure(2)
ezplot(g2)
 
%%%%%% Plot Density Difference   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
gDiff=g1-g2;
figure(3)
ezplot(gDiff, [0,2,0,1,-0.001,0.001])