function K=kernel(X,type,para)
N=size(X,1);

if strcmp(type,'gaussian')
    K=distanceMatrix(X).^2;
    K=exp(-K./para);
end
