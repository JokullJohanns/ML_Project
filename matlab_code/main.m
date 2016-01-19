clear all;
train_data=load('train_data.mat');
test_data=load('test.mat');

X_train=train_data.X;
X_test= test_data.gaussian;

%X_test=imnoise(X_train(2308,:),'salt & pepper', 0.2); %8 
%X_test=imnoise(X_train(2308,:),'gaussian', 0.25); %8 

%X_test=imnoise(X_train(1364,:),'salt & pepper', 0.2); %5
%X_test=imnoise(X_train(1364,:),'gaussian', 0.25); %5

%X_test=imnoise(X_train(1669,:),'salt & pepper', 0.2); %6
%X_test=imnoise(X_train(1669,:),'gaussian', 0.25); %6

%X_test=imnoise(X_train(1090,:),'gaussian', 0.25); %4
%X_test=imnoise(X_train(1090,:),'speckle', 0.2); %4

X_test=X_test(8,:);
X_test=imnoise(X_train(475,:),'gaussian',0, 0.25); %4
[eig_vec, eig_val, K]= kPCA(X_train, 64, 'gaussian', 100);  
z=kPCA_PreImage(X_test, eig_vec, X_train, 100);
img=reshape(z,[16,16]);
imshow(img',[]);