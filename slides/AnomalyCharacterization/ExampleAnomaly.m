clear all
close all
%Example Whispers 2012
%Santiago Velasco Forero
MU = [4 4; 10 10];
SIGMA = cat(3,[.1 0;0 .1],[.5 .1;.1 .5]);
iter=1;
i=.01;
p = [1-i,i];
obj = gmdistribution(MU,SIGMA,p);
ss=5000;
for i=1:100
    if i==1
        Y = random(obj,ss);
        t = Y;
        plot(Y(:,1),Y(:,2),'.');
        ylabel('Second Dimension')
        xlabel('First Dimension')
        grid on
        figure;
    else
        f=(t+rand(ss,2)./100);
        Y=[Y,f];
    end
end
%m=mean(Y);
%Yn=Y-repmat(m,30,1);
%S=inv(cov(Y));
%Yn=Y-repmat(m,50,1);
subplot(2,1,1)
tic
d=proj_depth_random(Y,100);
toc
%scatter3(Y(:,1),Y(:,2),d)
plot(Y(:,1),d/max(d),'.')
ylabel('Detector by Random Projections')
xlabel('First Dimension')
grid on
tic
%m=mean(Y);
%Yn=Y-repmat(m,ss,1);
res=mahal(Y,Y);
toc
%for i=1:ss
%res(i)=(Y(i,:)-m)*S*(Y(i,:)-m)';
%end

subplot(2,1,2)
%scatter3(Y(:,1),Y(:,2),res,'r')
plot(Y(:,1),res./max(res),'.r')
ylabel('Normalized Mahalanobis Distance')
xlabel('First Dimension')
grid on
min(res)