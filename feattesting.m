

imds = imageDatastore('five_classes_images', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

[imdsTrain,imdsTest] = splitEachLabel(imds,0.7,'randomized');

numImagesTrain = numel(imdsTrain.Labels);
idx = randperm(numImagesTrain,16);

I = imtile(imds, 'Frames', idx);

figure
imshow(I)
net = squeezenet;
inputSize = net.Layers(1).InputSize
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);

layer = 'pool10';
featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');
featuresTest = activations(net,augimdsTest,layer,'OutputAs','rows');
YTrain = imdsTrain.Labels;
YTest = imdsTest.Labels;
mdl = fitcecoc(featuresTrain,YTrain);
YPred = predict(mdl,featuresTest);
idx = [1 5 10 15];
figure
for i = 1:numel(idx)
    subplot(2,2,i)
    I = readimage(imdsTest,idx(i));
    label = YPred(idx(i));
    
    imshow(I)
    title(label)
end
accuracy = mean(YPred == YTest)