function [eigVector, eigValue, K] = kPCA(X,d,type,para)
N=size(X,1);
K0=kernel(X,type,para);
oneN=ones(N,N)/N;
K=K0-oneN*K0-K0*oneN+oneN*K0*oneN;

[V,D]=eig(K/N);
eigValue=diag(D);
[~,IX]=sort(eigValue,'descend');
eigVector=V(:,IX);
eigValue=eigValue(IX);

norm_eigVector=sqrt(sum(eigVector.^2));
eigVector=eigVector./repmat(norm_eigVector,size(eigVector,1),1);
eigVector=eigVector(:,1:d);