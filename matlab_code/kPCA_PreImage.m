function z=kPCA_PreImage(X_test,eigVector,X,para)
iter=1000;
N=size(X,1);
d=size(eigVector,2);

beta=zeros(d,1);
for i=1:d
    for j=1:N
           beta(i)= beta(i)+ eigVector(j,i)* exp(-norm(X_test-X(j,:))^2/para);
    end
end

y=beta
gamma=zeros(1,N);
for i=1:N
    gamma(i)=eigVector(i,1:d)*y;
end
z=mean(X)'; % initialization
for count=1:iter
    pre_z=z;
    xx=bsxfun(@minus,X',z);
    xx=-sum(xx.^2)/para;
    xx=exp(xx).*gamma;
    z=xx*X/sum(xx);
    z=z';
    if norm(pre_z-z)/norm(z)<0.00001
        break;
    end
end