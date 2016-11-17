% Learning an Attention Model in an Artificial Visual System
% Written by Alon Hazan, November 2016.
% contact: alonh dot tx at gmail dot com

function varargout = AVS(varargin)
% Begin - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @AVS_OpeningFcn, ...
    'gui_OutputFcn',  @AVS_OutputFcn, ...
    'gui_LayoutFcn',  [] , ...
    'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end

function varargout = AVS_OutputFcn(hObject, eventdata, handles)
varargout{1} = handles.output;
% End - DO NOT EDIT


function AVS_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;
DefaultProjectName = strrep(datestr(now),'-','');
DefaultProjectName = strrep(DefaultProjectName,' ','_');
set(handles.txtProjectName,'string',strrep(DefaultProjectName,':','')) %set default name for the project


guidata(hObject, handles);




% --- Executes on button press in cmdTrain.
function cmdTrain_Callback(hObject, eventdata, handles)
set(handles.cmdTrain,'enable','off')
set(handles.cmdStop,'enable','on')
global StopFlag
StopFlag=0;

while ~StopFlag
    IsContinue=1;
    TrainFlag=1;
    load('MNIST_train_all.mat') % contains MNIST dataset inside the matrix "I" and "labels"
    %   load('MNIST_Test_all.mat')
    
    
    % in case you wish to remove a certain digit from MNIST, this is the place to do it:
    indlablel = labels==0 | labels==1 | labels==2  | labels==3 | labels==4 | labels==5 | labels==6  | labels==7 | labels==8 | labels==9;
    I=I(:,:, indlablel );
    labels=labels(indlablel);
    TotalNumOfSamples=length(labels);
    Digits = unique(labels);
    NumberOfDigits = length(Digits);
    
    
    % Initialize
    DataFolderName=get(handles.txtProjectName,'string');
    cnuse=2;
    if ~exist(DataFolderName,'dir') || strcmp(DataFolderName,'temp')
        IsContinue=0; %create new project
    else
        for cn=2:400
            if exist([DataFolderName '_Con_' num2str(cn)] ,'dir')
                cnuse=cn+1;
            end
        end
        DataFolderName = [DataFolderName  '_Con_' num2str(cnuse)]; %create a continue folder for an existing project
    end
    mkdir(DataFolderName)
    
    Epochs=5*1000; % will run this many Epochs (use multiples of 1000)
    PreformanceEvalLength=0.01*Epochs; % number of epochs to look back at, in order to evaluate performance.
    Neurons=str2double(get(handles.txtNeurons,'string')); %Number of neurons in the network
    Glimpses=4; % number of glimpses per digit
    LearningRate=str2double(get(handles.txtLearningRate,'string'));
    LeakRate=1;
    InputFactor=0.3;
    ConnectedNeurons=10;%average number of connections to each neuron
    NoiseStd=0.5;
    FoveaRadius=3; % radius of the fovea
    Chaoticity=1;
    ImSize=28; % Zero padding MNIST images is needed only if we distracting overlay objects in the image (the original images are 28 by 28 pixels)
    DistractingObject=0; % add distracting block DistractingObject==1=fixed  DistractingObject==2=free (if you use a block, must choose ImSize to be 80 to allow room in the image)
    Guidance=0; % if 1, Enable demonstration (works only on the first chapter, not in continues folders)
    VariableGlimpses=0;
    CaseString = {'Guidance=1;','Guidance=0;'}; %compare the cases with/without Guidance to the attention learning.
    
    % save snapshot of project
    save([ DataFolderName  '/CaseString.mat'], 'CaseString');
    
    
    % Configure loop size
    cases=length(CaseString);
    tests=3;
    
    for Index =1:(cases*tests) % you may change "for" to "parfor"  in order to get parallelization
        [casenum,testnum]=ind2sub([cases,tests],Index);
        
        % Train network
        [net, storedata] = TrainTestAVS(I,labels,CaseString,DataFolderName,TotalNumOfSamples,TrainFlag,IsContinue,VariableGlimpses,Guidance,Chaoticity,ConnectedNeurons,NoiseStd,LearningRate,LeakRate,Neurons,PreformanceEvalLength,casenum,testnum,NumberOfDigits,Epochs,FoveaRadius,Glimpses,ImSize,InputFactor,DistractingObject,Digits,handles.txtMessage);
        
        if TrainFlag
            para_save([ DataFolderName  '/seq_storedata_C' num2str(casenum) '_T' num2str(testnum) ], storedata);
            para_save([ DataFolderName  '/seq_Network_C' num2str(casenum) '_T' num2str(testnum) ], net);
        end
        
        if isempty(getCurrentTask); %if not parallel (did not use "parfor")
            set(handles.txtMessage,'string',['Done with:  ' DataFolderName ':   Index ' num2str(Index) '/'  num2str(cases*tests)  ':   Start case: ' num2str(casenum) ',  test run: ' num2str(testnum)])
            Plot2GUI(handles.axes1,DataFolderName,CaseString)
            drawnow
        end
        
    end
    Plot2GUI(handles.axes1,DataFolderName,CaseString)
    set(handles.txtMessage,'string',['Done with:  ' DataFolderName])
    drawnow
end
set(handles.cmdTrain,'enable','on')

function  Plot2GUI(ax_handle,DataFolderName,CaseString)
cla(ax_handle,'reset')
axes(ax_handle)

StoreCases{length(CaseString)}=[];

str=DataFolderName;
ind=strfind(str,'_Con_');
if ~isempty(ind)
    str= str(1:ind-1);
end
listdirs= dir;
ind=zeros(length(listdirs),1);
for i=3:length(listdirs)
    if ~~strfind(listdirs(i).name,str)
        ind(i)=str2double(listdirs(i).name(length(str)+6:end));
    end
end
M=max(ind); %Number of continues found
D=dir([pwd '\' str]);
for i=3:length(D)
    if length(D(i).name) > 13 && strcmp(D(i).name(1:13),'seq_storedata')
        From=strfind(D(i).name,'_C')+2;
        To=strfind(D(i).name,'_T')-1;
        Ncase=str2double(D(i).name(From:To));
        StoreCases{Ncase}=[StoreCases{Ncase} ; importdata([str '\' D(i).name])];
    end
end

for m=2:M
    siz=size(StoreCases{Ncase},2);
    D=dir([ str '_Con_' num2str(m)]);
    for i=3:length(D)
        if length(D(i).name) > 13 && strcmp(D(i).name(1:13),'seq_storedata')
            From=strfind(D(i).name,'_C')+2;
            To=strfind(D(i).name,'_T')-1;
            Ncase=str2double(D(i).name(From:To));
            From=strfind(D(i).name,'_T')+2;
            To=strfind(D(i).name,'.')-1;
            Ntest=str2double(D(i).name(From:To));
            data = importdata(  [ str '_Con_' num2str(m) '\' D(i).name ]  );
            StoreCases{Ncase}(Ntest,siz+1:siz+length(data))  = data;
        end
        
    end
    
end

counter=1;
for i=1:length(CaseString)
    if ~isempty(StoreCases{i})
        Leg{counter}=CaseString{i};
        counter=counter+1;
    end
end

xlabel('Evaluation Intervals')
ylabel('Sucsses rate [%]')
grid on
hold on



for i=1:length(CaseString)
    if ~isempty(StoreCases{i})
        aa=gca;
        ind=aa.ColorOrderIndex;
        StoreCases{i}(StoreCases{i}==0)=nan;
        H{i}=plot(StoreCases{i}','color',aa.ColorOrder(ind,:));
        hline(i)=H{i}(1);
    end
end

legend(hline,Leg,'location','southeast')

function   [net, storedata] = TrainTestAVS(DataSet,labels,CaseString,DataFolderName,TotalNumOfSamples,TrainFlag,IsContinue,VariableGlimpses,Guidance,Chaoticity,ConnectedNeurons,NoiseStd,LearningRate,LeakRate,Neurons,PreformanceEvalLength,casenum,testnum,Nout_ID,Epochs,FoveaRadius,Glimpses,ImSize,InputFactor,DistractingObject,Digits,MessageHandle)

locfac=1.25; %factor between network output and image pixels
RMSprop=1; % adaptive learning rate method proposed by Geoff Hinton
E_Att=0;
E_ID=0;
counter=1;
storedata=zeros(1,Epochs/PreformanceEvalLength);

eval(CaseString{casenum}); % Overwrite parameters

if DistractingObject>0
    if ImSize<80
        display('ImSize size should be 80 or more for Block testing')
        return
    end
end

% Create RNN
Nin=3*(2*FoveaRadius-1)^2+1; %number of input neurons (cells in the eye)

if TrainFlag
    if IsContinue % load existing network to cintinue learning
        Guidance=0; % override use of Guidance in continued learning, we only use it in the first chapter.
        MNISTindex= findstr(DataFolderName,'_Con_');
        num = str2double(DataFolderName(MNISTindex+5:end));
        if num==2
            OriginalDir=DataFolderName(1:(MNISTindex-1));
        else
            OriginalDir=[DataFolderName(1:(MNISTindex-1)) '_Con_' num2str(num-1)];
        end
        
        filename=[OriginalDir  '/seq_Network_C' num2str(casenum) '_T' num2str(testnum) '.mat'];
        
        if exist(filename,'file')
            net=importdata(filename);
            if isempty(net)
                storedata=[];
                return
            end
        else
            net=[]; storedata=[];
            return
        end
    else % create new network
        net = CreateNeuralNetwork(Neurons,Nin,InputFactor,Nout_ID,ConnectedNeurons,Chaoticity,NoiseStd,LearningRate,LeakRate);
    end
    
else % load existing network for testing
    filename=['seq_Network_C' num2str(casenum) '_t' num2str(testnum) '.mat'];
    if exist(filename,'file')
        net=importdata(filename);
    else
        net=[];  storedata=[];
        return
    end
end

EpochSize=40;
AttentionNoise(2,Glimpses,EpochSize)=0;
IDNoise(net.Nout_ID,Glimpses,EpochSize)=0;
GoStopNoise(2,Glimpses,EpochSize)=0;
PerformanceLog = zeros(1,PreformanceEvalLength);
Reward=zeros(1,EpochSize);

if ~TrainFlag
    % in Test mode go over 2 times the number of samples in the dataset (does not strictly go over all samples but empiricly the result is very close)
    Epochs=round(2*TotalNumOfSamples/EpochSize);
    PreformanceEvalLength=Epochs;
end


for epoch=1:Epochs
    
    Grad_Sum_Attention=zeros(2,Neurons,EpochSize);
    Grad_Sum_ID=zeros(net.Nout_ID,Neurons,EpochSize);
    Grad_Sum_GoStop=zeros(2,Neurons,EpochSize);
    net.activation=zeros(Neurons,Glimpses+1,EpochSize);
    GlimpsesCounter=Glimpses*ones(1,EpochSize);
    
    for Sample=1:EpochSize
        
        MNISTindex=randi([1 TotalNumOfSamples],1,1);
        [ImageSample, x_digit_center, y_digit_center] = RandomizeMnistSample(DataSet(:,:,MNISTindex),ImSize,zeros(ImSize),DistractingObject);
        CurrentPosition = GoodFirstGlimpse(ImageSample,ImSize,FoveaRadius);
        
        
        for glimpse=1:Glimpses
            
            Observation = TakeAGlimpse(ImageSample,ImSize,CurrentPosition,FoveaRadius);
            net.activation(:,glimpse+1,Sample) =  (1-net.LeakRate).*net.activation(:,glimpse,Sample)+   net.LeakRate*(net.W*tanh(net.activation(:,glimpse,Sample)) + net.Win*Observation ); % Network Step
            
            if TrainFlag
                
                if Guidance && glimpse==(Glimpses-1) && ~mod(Sample,10) && epoch<(0.8*Epochs) % Guidance
                    AttentionOutput =([x_digit_center;y_digit_center]-CurrentPosition)/locfac; % Demonstrate Attention output
                    AttentionNoise(:,glimpse,Sample) = AttentionOutput -  net.Att_Wout*tanh(net.activation(:,glimpse+1,Sample));
                else %regular training
                    AttentionNoise(:,glimpse,Sample) = net.NoiseStd.*randn(2,1);
                    AttentionOutput = net.Att_Wout*tanh(net.activation(:,glimpse+1,Sample)) +  AttentionNoise(:,glimpse,Sample); % Update output
                end
                
                if VariableGlimpses
                    GoStopNoise(:,glimpse,Sample) = net.NoiseStd.*randn(2,1);
                    GoStopOutput = net.GoStop_Wout*tanh(net.activation(:,glimpse+1,Sample)) +  GoStopNoise(:,glimpse,Sample); % Update output
                end
                
            else % Test
                
                AttentionOutput = net.Att_Wout*tanh(net.activation(:,glimpse+1,Sample));
                GoStopOutput = net.GoStop_Wout*tanh(net.activation(:,glimpse+1,Sample));
                
            end
            
            CurrentPosition = CurrentPosition+locfac*AttentionOutput; %Update Position based on Attention output
            
            if VariableGlimpses
                [~,gostopval]=max(GoStopOutput);
                if gostopval==2 % stop glimpsing
                    GlimpsesCounter(Sample)=glimpse;
                    break
                end
            end
            
        end
        
        
        if TrainFlag
            IDNoise(:,Sample) = net.NoiseStd.*randn(net.Nout_ID,1);
            IDOutput = net.ID_Wout*tanh(net.activation(:,GlimpsesCounter(Sample)+1,Sample)) + IDNoise(:,Sample); % Update output
        else
            IDOutput = net.ID_Wout*tanh(net.activation(:,GlimpsesCounter(Sample)+1,Sample)); % Update output
        end
        
        % Calc Reward for the last trajectory
        [~,WinnerNeuron]=max(IDOutput); %winner takes all
        if VariableGlimpses
            Reward(Sample)=Digits(WinnerNeuron)==labels(MNISTindex) .* (1 - 0.01*GlimpsesCounter(Sample));
            BinaryReward(Sample)=Digits(WinnerNeuron)==labels(MNISTindex);
        else
            Reward(Sample)=Digits(WinnerNeuron)==labels(MNISTindex);
            BinaryReward(Sample)=Reward(Sample);
        end
    end
    
    
    if TrainFlag
        
        AttentionNoise=AttentionNoise./(net.NoiseStd.^2);
        IDNoise=IDNoise./(net.NoiseStd.^2);
        net.activation=tanh(net.activation);
        
        
        for Sample=1:EpochSize
            Grad_Sum_Attention(:,:,Sample) = AttentionNoise(:,1:GlimpsesCounter(Sample),Sample) * net.activation(:,2:GlimpsesCounter(Sample)+1,Sample).';
            Grad_Sum_ID(:,:,Sample)=(IDNoise(:,Sample) *  net.activation(:,GlimpsesCounter(Sample)+1,Sample).');
            if VariableGlimpses
                Grad_Sum_GoStop(:,:,Sample)=(GoStopNoise(:,Sample) *  net.activation(:,GlimpsesCounter(Sample)+1,Sample).');
            end
        end
        
        Reinforce_Numerator_Att=sum(  bsxfun(@times, (Grad_Sum_Attention.^2),permute(Reward,[3 1 2])) ,3 )./EpochSize;
        Reinforce_Denominator_Att=sum((Grad_Sum_Attention).^2,3)./EpochSize;
        Reinforce_Att=repmat(Reinforce_Numerator_Att./Reinforce_Denominator_Att,1,1,EpochSize) ;
        Effective_Reward_Att=bsxfun(@minus,permute(Reward,[3 1 2]),Reinforce_Att);
        Grad_Sum_Attention=sum( Grad_Sum_Attention.*Effective_Reward_Att, 3 );
        
        if RMSprop
            E_Att = 0.9.*E_Att + 0.1*(Grad_Sum_Attention./EpochSize).^2;
            net.Att_Wout = net.Att_Wout + 0.1*net.LearningRate.*Grad_Sum_Attention./EpochSize./sqrt(E_Att+100*eps(E_Att));
        else
            net.Att_Wout = net.Att_Wout + net.LearningRate.*Grad_Sum_Attention./EpochSize;
        end
        
        Reinforce_Numerator_ID=sum(  bsxfun(@times, (Grad_Sum_ID.^2),permute(Reward,[3 1 2])) ,3 )./EpochSize;
        Reinforce_Denominator_ID=sum((Grad_Sum_ID).^2,3)./EpochSize;
        Reinforce_ID=repmat(Reinforce_Numerator_ID./Reinforce_Denominator_ID,1,1,EpochSize) ;
        Effective_Reward_ID=bsxfun(@minus,permute(Reward,[3 1 2]),Reinforce_ID);
        Grad_Sum_ID=sum( Grad_Sum_ID.*Effective_Reward_ID, 3 );
        
        if RMSprop
            E_ID = 0.9.*E_ID + 0.1*(Grad_Sum_ID./EpochSize).^2;
            net.ID_Wout = net.ID_Wout + 0.1*net.LearningRate.*Grad_Sum_ID./EpochSize./sqrt(E_ID+10*eps);
        else
            net.ID_Wout = net.ID_Wout + net.LearningRate.*Grad_Sum_ID./EpochSize;
        end
        
        if VariableGlimpses
            Reinforce_Numerator_GoStop=sum(  bsxfun(@times, (Grad_Sum_GoStop.^2),permute(Reward,[3 1 2])) ,3 )./EpochSize;
            Reinforce_Denominator_GoStop=sum((Grad_Sum_GoStop).^2,3)./EpochSize;
            Reinforce_GoStop=repmat(Reinforce_Numerator_GoStop./Reinforce_Denominator_GoStop,1,1,EpochSize) ;
            Effective_Reward_GoStop=bsxfun(@minus,permute(Reward,[3 1 2]),Reinforce_GoStop);
            Grad_Sum_GoStop=sum( Grad_Sum_GoStop.*Effective_Reward_GoStop, 3 );
            
            if RMSprop
                E_GoStop = 0.9.*E_GoStop + 0.1*(Grad_Sum_GoStop./EpochSize).^2;
                net.GoStop_Wout = net.GoStop_Wout + 0.1*net.LearningRate.*Grad_Sum_GoStop./EpochSize./sqrt(E_GoStop+10*eps);
            else
                net.GoStop_Wout = net.GoStop_Wout + neta.eta.*Grad_Sum_GoStop./EpochSize;
            end
            
        end
        
    end
    
    
    PerformanceLog=circshift(PerformanceLog,[0 1]);
    PerformanceLog(:,1)=sum(BinaryReward);
    
    if ~mod(epoch,PreformanceEvalLength)
        summem=sum(PerformanceLog)/PreformanceEvalLength/EpochSize*100;
        storedata(1,counter)=summem;
        counter=counter+1;
        
        if ~mod(epoch,PreformanceEvalLength*10)
            set(MessageHandle,'string',[DataFolderName  '  C#'  num2str(casenum) '   T#' num2str(testnum)    '   Progress: ' num2str(round(1000*counter/size(storedata,2))/10) '%' ...
                '   Success rate: ' num2str(round(summem*100)/100) '%'  ])
            drawnow
        end
        
    end
end

function net = CreateNeuralNetwork(N,Nin,InputFactor,Nout_ID,ConnectedNeurons,Chaoticity,NoiseStd,LearningRate,LeakRate)
Sparsity=ConnectedNeurons./N;
net.N=N;
net.Nin=Nin;
net.Nout_ID=Nout_ID;
net.Sparsity=Sparsity;
net.LeakRate = LeakRate;
net.Chaoticity = Chaoticity;
net.NoiseStd = NoiseStd;
net.LearningRate = LearningRate;
inputneurons=N;

% Create a full random recurrent connections matrix N*N:
net.W = randn(N)./sqrt(Sparsity*N);
% Sparse the recurrent connections matrix:
net.W(binornd(1,1-Sparsity,N,N)>0.5)=0;
net.W=sparse(net.W);

% Create uniform random connections for input & feedback
net.Win = -1 + 2*rand(N,Nin);
net.Win((inputneurons+1):N,:)=0;
net.Win = InputFactor*net.Win;

% Create output weights
net.Att_Wout = zeros(2,N);
net.ID_Wout = zeros(Nout_ID,N);
net.GoStop_Wout = zeros(2,N);


function CurrentPosition = GoodFirstGlimpse(ImgeSamples,ImSize,FoveaRadius)
CurrentPosition=round((rand(2,1)*ImSize - ImSize/2).*0.95); %random position
Rmax=4*(FoveaRadius-1);
Sample=1;
while 1
    X=round((-Rmax:Rmax)+CurrentPosition(1,Sample)+ImSize/2);
    Y=round((-Rmax:Rmax)+CurrentPosition(2,Sample)+ImSize/2);
    X(X>ImSize)=ImSize;
    X(X<1)=1;
    Y(Y>ImSize)=ImSize;
    Y(Y<1)=1;
    if sum(sum(ImgeSamples(X,Y)))>5; %first glimpse must observe at least a small part of the digit
        return
    else
        CurrentPosition=round((rand(2,1)*ImSize - ImSize/2).*0.95); %random position
    end
end


function [FullImage, x, y]= RandomizeMnistSample(OriginalImage,ImSize,FullImage,DistractingObject)
blksiz=8;
Padding=(ImSize-28)/2;
m=round(ImSize/2)-14+1;
n=m;
FullImage(m:m+27,n:n+27)=OriginalImage(:,:); %the digit is at the center of the padded image (the AVS has no sense of absolute position so this is not informative to the solution)
if DistractingObject==1 %distracting object with fixed position
    FullImage(m+9:m+19,n+32:n+42)=1;
elseif DistractingObject==2 %distracting object with random position
    seed=randi(4,1);
    switch seed
        case 1
            bm=randi(Padding-blksiz,1);
            bn=randi(ImSize-blksiz,1);
        case 2
            bm=randi([ImSize-Padding,ImSize-blksiz],1);
            bn=randi(ImSize-blksiz,1);
        case 3
            bm=randi([Padding,ImSize-blksiz],1);
            bn=randi([Padding+28,ImSize-blksiz],1);
        case 4
            bm=randi([Padding,ImSize-blksiz],1);
            bn=randi([1,Padding-blksiz],1);
    end
    FullImage(bm:bm+blksiz,bn:bn+blksiz)=1;
end
y=(n+14-ImSize/2);
x=(m+14-ImSize/2);


function  Observation = TakeAGlimpse(InputImage,ImSize,CurrentPosition,FoveaRadius)

Nin=3*(2*FoveaRadius-1)^2+1;
Observation=zeros(Nin,1);
Observation(1,:)=1; %input bias

FoveaRadius=FoveaRadius-1;
FoveaSize=2*FoveaRadius+1;
ObservMat=zeros(FoveaSize,FoveaSize,3);


X=round((-FoveaRadius:FoveaRadius)+CurrentPosition(1)+ImSize/2);
Y=round((-FoveaRadius:FoveaRadius)+CurrentPosition(2)+ImSize/2);
X(X>ImSize)=ImSize;
X(X<1)=1;
Y(Y>ImSize)=ImSize;
Y(Y<1)=1;
ObservMat(:,:,1)=InputImage(X,Y);
peripheral=2;
for dsemp=[2,4]
    X=round((-FoveaRadius*dsemp:FoveaRadius*dsemp)+CurrentPosition(1)+ImSize/2);
    Y=round((-FoveaRadius*dsemp:FoveaRadius*dsemp)+CurrentPosition(2)+ImSize/2);
    X(X>ImSize)=ImSize;
    X(X<1)=1;
    Y(Y>ImSize)=ImSize;
    Y(Y<1)=1;
    ObservMat(:,:,peripheral)=downsample(downsample(InputImage(X,Y),dsemp)',dsemp)';
    peripheral=peripheral+1;
end

Observation(2:end,:)=reshape(ObservMat,Nin-1,1);





function cmdStop_Callback(hObject, eventdata, handles)
set(handles.cmdStop,'enable','off')
msgbox('Will stop after this chapter ends')
global StopFlag
StopFlag=1;



function txtProjectName_Callback(hObject, eventdata, handles)
function txtProjectName_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
function figure1_CreateFcn(hObject, eventdata, handles)
function txtNeurons_Callback(hObject, eventdata, handles)
function txtNeurons_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
function txtLearningRate_Callback(hObject, eventdata, handles)
function txtLearningRate_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
function para_save(fname, x)
save(fname, 'x')
